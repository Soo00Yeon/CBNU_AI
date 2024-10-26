import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import sim  # CoppeliaSim Python API
import random
import time

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
memory_size = 100000
episodes = 50

def create_q_model():
    inputs = layers.Input(shape=(256, 256, 3,))
    layer1 = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    layer2 = layers.MaxPooling2D((2, 2))(layer1)
    layer3 = layers.Conv2D(64, (3, 3), activation="relu")(layer2)
    layer4 = layers.MaxPooling2D((2, 2))(layer3)
    layer5 = layers.Flatten()(layer4)
    layer6 = layers.Dense(128, activation="relu")(layer5)
    action = layers.Dense(4, activation="linear")(layer6)

    model = tf.keras.Model(inputs=inputs, outputs=action)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model

class Memory:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def connect_to_simulator():
    sim.simxFinish(-1)
    client_id = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
    if client_id != -1:
        print('Connected to remote API server')
        return client_id
    else:
        print('Failed connecting to remote API server')
        return None

def initialize_simulation(client_id):
    sim.simxStartSimulation(client_id, sim.simx_opmode_oneshot)
    time.sleep(1.0)

def terminate_simulation(client_id):
    sim.simxStopSimulation(client_id, sim.simx_opmode_oneshot)
    time.sleep(1.0)

def get_state(client_id, vision_sensor_handle):
    sim.simxGetVisionSensorImage(client_id, vision_sensor_handle, 0, sim.simx_opmode_streaming)
    time.sleep(1.0)
    for attempt in range(30):
        error_code, resolution, image = sim.simxGetVisionSensorImage(client_id, vision_sensor_handle, 0, sim.simx_opmode_buffer)
        if error_code == sim.simx_return_ok:
            break
        time.sleep(0.3)

    if error_code != sim.simx_return_ok:
        print(f"Failed to get image, error code: {error_code}")
        return None

    print(f"Resolution: {resolution}")
    print(f"Image length: {len(image)}")

    if len(resolution) < 2 or len(image) == 0:
        print("Failed to get valid image or resolution from vision sensor")
        return None

    # 이미지를 TensorFlow 텐서로 변환
    image = tf.convert_to_tensor(image, dtype=tf.int32)
    image = tf.clip_by_value(image, 0, 255)  # 값을 0과 255 사이로 제한
    image = tf.cast(image, tf.uint8)  # uint8로 캐스팅

    # 이미지의 크기가 256x256이 아니면 크기를 변경
    if resolution != [256, 256]:
        image = tf.reshape(image, [resolution[1], resolution[0], 3])  # 원래 이미지 크기로 변경
        image = tf.image.resize(image, [256, 256])  # 크기 조정
    else:
        image = tf.reshape(image, [256, 256, 3])  # 크기 조정 불필요 시 바로 reshape

    # 이미지를 float32로 캐스팅하고 정규화
    image = tf.cast(image, tf.float32) / 255.0

    return image

def execute_action(client_id, action, joint3_handle, joint4_handle):
    _, joint3_position = sim.simxGetJointPosition(client_id, joint3_handle, sim.simx_opmode_blocking)
    _, joint4_position = sim.simxGetJointPosition(client_id, joint4_handle, sim.simx_opmode_blocking)
    joint_movement_increment = np.deg2rad(10)  # 이동 증분을 조정하여 더 세밀한 조작 가능
    if action == 0:
        new_joint3_position = joint3_position + joint_movement_increment
        sim.simxSetJointTargetPosition(client_id, joint3_handle, new_joint3_position, sim.simx_opmode_oneshot)
    elif action == 1:
        new_joint3_position = joint3_position - joint_movement_increment
        sim.simxSetJointTargetPosition(client_id, joint3_handle, new_joint3_position, sim.simx_opmode_oneshot)
    elif action == 2:
        new_joint4_position = joint4_position + joint_movement_increment
        sim.simxSetJointTargetPosition(client_id, joint4_handle, new_joint4_position, sim.simx_opmode_oneshot)
    elif action == 3:
        new_joint4_position = joint4_position - joint_movement_increment
        sim.simxSetJointTargetPosition(client_id, joint4_handle, new_joint4_position, sim.simx_opmode_oneshot)

def reset_softbody_position(client_id, softbody_handle):
    random_x = -1.001
    random_y = np.random.uniform(-1.3, 0.5)
    random_z = np.random.uniform(0.2, 1)
    sim.simxSetObjectPosition(client_id, softbody_handle, -1, [random_x, random_y, random_z], sim.simx_opmode_oneshot)

def compute_reward(state, softbody_position, vision_sensor_position):
    center = (0.5, 0.5)
    softbody_center = (
        (softbody_position[0] + vision_sensor_position[0]) / (2 * vision_sensor_position[0]),
        (softbody_position[1] + vision_sensor_position[1]) / (2 * vision_sensor_position[1])
    )
    distance = np.sqrt((softbody_center[0] - center[0]) ** 2 + (softbody_center[1] - center[1]) ** 2)
    reward = -distance
    print(f"Reward: {reward}, Distance: {distance}")
    return reward

def check_termination(state, softbody_position, vision_sensor_position):
    center = (0.5, 0.5)
    softbody_center = (
        (softbody_position[0] + vision_sensor_position[0]) / (2 * vision_sensor_position[0]),
        (softbody_position[1] + vision_sensor_position[1]) / (2 * vision_sensor_position[1])
    )
    distance = np.sqrt((softbody_center[0] - center[0]) ** 2 + (softbody_center[1] - center[1]) ** 2)
    if distance < 0.1:  # 중앙에 가까운지 확인하는 임계값 설정
        return True
    return False

def train_dqn():
    global epsilon

    client_id = connect_to_simulator()
    if client_id is None:
        return

    initialize_simulation(client_id)

    _, vision_sensor_handle = sim.simxGetObjectHandle(client_id, 'Vision_sensor', sim.simx_opmode_blocking)
    _, joint3_handle = sim.simxGetObjectHandle(client_id, 'UR5_joint3', sim.simx_opmode_blocking)
    _, joint4_handle = sim.simxGetObjectHandle(client_id, 'UR5_joint4', sim.simx_opmode_blocking)
    _, softbody_handle = sim.simxGetObjectHandle(client_id, 'softBody', sim.simx_opmode_blocking)

    model = create_q_model()
    target_model = create_q_model()
    target_model.set_weights(model.get_weights())

    memory = Memory(memory_size)

    for episode in range(episodes):
        reset_softbody_position(client_id, softbody_handle)
        state = get_state(client_id, vision_sensor_handle)
        if state is None:
            continue
        done = False
        total_reward = 0

        step = 0  # step 변수를 초기화
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(20)
            else:
                q_values = model.predict(np.expand_dims(state, axis=0))
                action = np.argmax(q_values[0])

            execute_action(client_id, action, joint3_handle, joint4_handle)
            next_state = get_state(client_id, vision_sensor_handle)
            if next_state is None:
                break

            _, softbody_position = sim.simxGetObjectPosition(client_id, softbody_handle, vision_sensor_handle, sim.simx_opmode_blocking)
            _, vision_sensor_position = sim.simxGetObjectPosition(client_id, vision_sensor_handle, -1, sim.simx_opmode_blocking)

            reward = compute_reward(next_state, softbody_position, vision_sensor_position)
            done = check_termination(next_state, softbody_position, vision_sensor_position)

            memory.add((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            step += 1  # step 변수 증가

            if len(memory.buffer) > batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                q_values_next = target_model.predict(next_states)
                targets = rewards + (1 - dones) * gamma * np.amax(q_values_next, axis=1)

                q_values = model.predict(states)
                for i, action in enumerate(actions):
                    q_values[i][action] = targets[i]

                model.fit(states, q_values, epochs=1, verbose=0)

            if step % 10 == 0:
                target_model.set_weights(model.get_weights())

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    terminate_simulation(client_id)
    sim.simxFinish(client_id)

if __name__ == "__main__":
    train_dqn()
