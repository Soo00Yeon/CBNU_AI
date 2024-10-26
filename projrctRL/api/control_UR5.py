import sim
import time
import sys

print("program started")
sim.simxFinish(-1)
clintID = sim.simxStart('127.0.01',19999,True,True,5000,5)

if(clintID != -1):
    print('Connected Sucessfully')
else:
    sys.exit('Faild Connected')


time.sleep(1)

error_code, first_joint_handle = sim.simxGetObjectHandle(clintID, 'UR5/link/joint/lint/joint',sim.simx_opmode_oneshot_wait)
error_code, second_joint_handle = sim.simxGetObjectHandle(clintID, 'UR5/link/joint/lint/joint/lint/joint',sim.simx_opmode_oneshot_wait)

error_code = sim.simxSetJointTargetVelocity(clintID, first_joint_handle, 0.2,sim.simx_opmode_oneshot_wait)
error_code = sim.simxSetJointTargetVelocity(clintID, second_joint_handle, 0.2,sim.simx_opmode_oneshot_wait)
