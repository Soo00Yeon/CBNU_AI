from PIL import Image
import numpy as np
from pathlib import Path
import random
import json
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import traceback
from typing import List, Tuple, Optional
import cv2
from filterpy.kalman import KalmanFilter

# ------------------------
# 1) 모델 로딩 및 개선된 추론 클래스
# ------------------------

def compute_iou(box1: List[float], box2: List[float]) -> float:
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def get_organ_detector_model(num_classes: int = 2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

class OrganTracker:
    def __init__(self, model_path: str, device):
        self.device = device
        self.model = get_organ_detector_model()
        # [수정 1] weights_only=True 추가하여 경고 해결
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(self.device).eval()
        self.transform = T.Compose([T.ToTensor()])

    @torch.no_grad()
    def detect_organ(self, image_path: Path) -> Optional[List[Tuple[List[float], float]]]:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        prediction = self.model(image_tensor)[0]
        
        if 'scores' not in prediction or not prediction['scores'].numel():
            return None

        scores = prediction['scores'].cpu().numpy()
        boxes = prediction['boxes'].cpu().numpy()
        
        top_indices = np.argsort(scores)[-5:][::-1]
        detections = [(boxes[i].tolist(), float(scores[i])) for i in top_indices]
        return detections


# ------------------------
# 2) IoU 기반 평가 로직이 적용된 함수
# ------------------------
def generate_motion_test_data(
    original_image_path: str,
    tracker: OrganTracker,
    output_dir: str = "test_data_final2_1_2",
    num_samples: int = 10,
    iou_threshold: float = 0.3  # IoU 매칭을 위한 최소 임계값
):
    output_path = Path(output_dir); output_path.mkdir(exist_ok=True)
    img = Image.open(original_image_path).convert("RGB")
    w, h = img.size

    base_img_path = output_path / "0_original.png"
    img.save(base_img_path)

    # 원본 이미지에서 가장 점수 높은 객체를 기준으로 설정
    base_detections = tracker.detect_organ(base_img_path)
    if not base_detections:
        print("오류: 원본 이미지에서 기준 객체를 찾지 못했습니다.")
        return []
    base_box = base_detections[0][0] # 가장 점수 높은 박스를 기준으로 사용

    results = []
    skipped_count = 0
    for i in range(num_samples):
        gt_scale = random.uniform(0.8, 1.2)
        gt_dx = random.randint(-int(w * 0.1), int(w * 0.1)) # 이동 범위 약간 축소
        gt_dy = random.randint(-int(h * 0.1), int(h * 0.1))

        # 이미지 변환
        sw, sh = int(w * gt_scale), int(h * gt_scale)
        resized = img.resize((sw, sh), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        paste_x = (w - sw) // 2 + gt_dx
        paste_y = (h - sh) // 2 + gt_dy
        canvas.paste(resized, (paste_x, paste_y))
        tpath = output_path / f"{i+1}_transformed.png"
        canvas.save(tpath)

        # 변환된 이미지에서 후보 객체들 탐지
        transformed_detections = tracker.detect_organ(tpath)
        if not transformed_detections:
            print(f"알림: {tpath.name}에서 객체 탐지 실패. 건너뜁니다.")
            skipped_count += 1
            continue

        # IoU 기반으로 가장 적합한 객체 찾기
        best_match_box = None
        max_iou = -1
        for t_box, t_score in transformed_detections:
            iou = compute_iou(base_box, t_box)
            if iou > max_iou:
                max_iou = iou
                best_match_box = t_box
        
        # 찾은 객체가 기준 IoU를 넘지 못하면 추적 실패로 간주
        if max_iou < iou_threshold:
            print(f"알림: {tpath.name}에서 추적 실패 (최대 IoU: {max_iou:.2f}). 건너뜁니다.")
            skipped_count += 1
            continue

        # 예측값 계산
        prev_cx, prev_cy = (base_box[0] + base_box[2]) / 2, (base_box[1] + base_box[3]) / 2
        prev_area = max(1e-6, (base_box[2] - base_box[0]) * (base_box[3] - base_box[1]))
        curr_cx, curr_cy = (best_match_box[0] + best_match_box[2]) / 2, (best_match_box[1] + best_match_box[3]) / 2
        curr_area = max(1e-6, (best_match_box[2] - best_match_box[0]) * (best_match_box[3] - best_match_box[1]))

        pred_dx, pred_dy = curr_cx - prev_cx, curr_cy - prev_cy
        pred_scale = float(np.sqrt(curr_area / prev_area))

        results.append({
            "sample_index": i + 1,
            "ground_truth": {"dx": gt_dx, "dy": gt_dy, "scale": gt_scale},
            "predicted": {"dx": pred_dx, "dy": pred_dy, "scale": pred_scale},
            "error": {"dx_error": abs(gt_dx - pred_dx), "dy_error": abs(gt_dy - pred_dy), "scale_error": abs(gt_scale - pred_scale)},
            "iou_matched": max_iou
        })
    
    print(f"\n평가 완료: {num_samples}개 중 {num_samples - skipped_count}개 성공, {skipped_count}개 실패/건너뜀.")
    return results

import cv2
import numpy as np

def refine_flow(prev_gray: np.ndarray, curr_gray: np.ndarray, prev_box: List[float]) -> Tuple[float, float]:
    """ [수정 2] Optical Flow 계산 방식을 전체 프레임 기준으로 변경 """
    mask = np.zeros_like(prev_gray)
    x1, y1, x2, y2 = map(int, prev_box)
    mask[y1:y2, x1:x2] = 255
    
    feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)

    if prev_pts is None:
        return 0.0, 0.0

    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

    if curr_pts is None or st is None:
        return 0.0, 0.0

    good_prev = prev_pts[st == 1]
    good_curr = curr_pts[st == 1]

    if len(good_prev) < 5:
        return 0.0, 0.0
    
    flow = good_curr - good_prev
    dx, dy = np.mean(flow, axis=0)
    return float(dx), float(dy)

def init_kalman_filter(x: float, y: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]) # 상태 전이
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # 관측
    kf.P *= 1000.
    kf.R = np.eye(2) * 5 # 측정 노이즈
    kf.x = np.array([x, y, 0., 0.])
    return kf

from filterpy.kalman import KalmanFilter
import numpy as np

def init_kalman(x0, y0):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    kf.P *= 1000.0
    kf.R = np.eye(2)*5.0
    kf.Q = np.eye(4)*0.1
    kf.x = np.array([x0,y0,0.0,0.0], dtype=float)
    return kf

def evaluate_with_improvements(
    original_image_path: str,
    tracker: OrganTracker,
    output_dir: str = "test_data_final2_1_2",
    num_samples: int = 20,
    iou_threshold: float = 0.3
):
    output_path = Path(output_dir); output_path.mkdir(exist_ok=True)
    base_img = Image.open(original_image_path).convert("RGB")
    w, h = base_img.size

    base_img_path = output_path / "0_original.png"; base_img.save(base_img_path)

    base_detections = tracker.detect_organ(base_img_path)
    if not base_detections:
        print("오류: 원본 이미지에서 기준 객체를 찾지 못했습니다.")
        return []
    
    base_box = base_detections[0][0]
    prev_cx, prev_cy = (base_box[0] + base_box[2]) / 2, (base_box[1] + base_box[3]) / 2
    kf = init_kalman_filter(prev_cx, prev_cy)

    results = []
    skipped_count = 0
    prev_gray = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2GRAY)

    for i in range(num_samples):
        gt_scale = random.uniform(0.8, 1.2); gt_dx = random.randint(-int(w*0.1), int(w*0.1)); gt_dy = random.randint(-int(h*0.1), int(h*0.1))

        # 이미지 변환
        sw, sh = int(w * gt_scale), int(h * gt_scale)
        resized = base_img.resize((sw, sh), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0,0,0))
        paste_x, paste_y = (w - sw) // 2 + gt_dx, (h - sh) // 2 + gt_dy
        canvas.paste(resized, (paste_x, paste_y))
        tpath = output_path / f"{i+1}_transformed.png"; canvas.save(tpath)

        # IoU 기반 매칭
        detections = tracker.detect_organ(tpath)
        if not detections:
            skipped_count += 1; continue
        
        max_iou, best_box = -1, None
        for t_box, _ in detections:
            iou = compute_iou(base_box, t_box)
            if iou > max_iou: max_iou, best_box = iou, t_box
        
        if max_iou < iou_threshold:
            skipped_count += 1; continue
        
        # Optical Flow 및 Kalman Filter 적용
        curr_gray = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2GRAY)
        dx_flow, dy_flow = refine_flow(prev_gray, curr_gray, base_box)
        
        # 칼만 필터 예측 및 업데이트
        kf.predict()
        # 측정값은 (탐지된 박스 중심 + flow 보정값)
        
        measurement = np.array([(best_box[0] + best_box[2])/2 + dx_flow, (best_box[1] + best_box[3])/2 + dy_flow])
        # dx_flow, dy_flow = refine_flow(...) # 이 라인 주석 처리
        # measurement = np.array([(best_box[0] + best_box[2])/2 + dx_flow, ...])
        #measurement = np.array([(best_box[0] + best_box[2])/2, (best_box[1] + best_box[3])/2]) # flow 보정 없이 사용
        kf.update(measurement)
        
        pred_cx, pred_cy = kf.x[0], kf.x[1]
        pred_dx, pred_dy = pred_cx - prev_cx, pred_cy - prev_cy

        # Scale 계산
        prev_area = max(1e-6, (base_box[2]-base_box[0]) * (base_box[3]-base_box[1]))
        curr_area = max(1e-6, (best_box[2]-best_box[0]) * (best_box[3]-best_box[1]))
        pred_scale = float(np.sqrt(curr_area / prev_area))
        
        results.append({
            "sample_index": i + 1,
            "ground_truth": {"dx": gt_dx, "dy": gt_dy, "scale": gt_scale},
            "predicted": {"dx": pred_dx, "dy": pred_dy, "scale": pred_scale},
            "error": {"dx_error": abs(gt_dx - pred_dx), "dy_error": abs(gt_dy - pred_dy), "scale_error": abs(gt_scale - pred_scale)},
            "iou_matched": max_iou
        })
        
        prev_gray, base_box = curr_gray, best_box
        prev_cx, prev_cy = pred_cx, pred_cy

    print(f"\n평가 완료: {num_samples}개 중 {len(results)}개 성공, {skipped_count}개 실패/건너뜀.")
    return results

# ------------------------
# 3) 요약 지표 및 JSON 저장 함수 (동일)
# ------------------------
def summarize_metrics(results):
    import numpy as np
    dx_err = np.array([r["error"]["dx_error"] for r in results], float)
    dy_err = np.array([r["error"]["dy_error"] for r in results], float)
    sc_err = np.array([r["error"]["scale_error"] for r in results], float)
    mae = {"dx": float(dx_err.mean()), "dy": float(dy_err.mean()), "scale": float(sc_err.mean())}
    rmse = {"dx": float(np.sqrt((dx_err**2).mean())),
            "dy": float(np.sqrt((dy_err**2).mean())),
            "scale": float(np.sqrt((sc_err**2).mean()))}
    dx_gt = np.array([r["ground_truth"]["dx"] for r in results], float)
    dy_gt = np.array([r["ground_truth"]["dy"] for r in results], float)
    sc_gt = np.array([r["ground_truth"]["scale"] for r in results], float)
    def acc(gt, err):
        denom = float(np.mean(np.abs(gt))); denom = denom if denom>=1e-6 else 1e-6
        val = 100.0 * (1.0 - float(np.mean(np.abs(err)))/denom)
        return max(0.0, val)
    return {"MAE": mae, "RMSE": rmse,
            "Accuracy_percent": {"dx": acc(dx_gt, dx_err), "dy": acc(dy_gt, dy_err), "scale": acc(sc_gt, sc_err)}}

def to_py(obj):
    import numpy as np
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: to_py(v) for k,v in obj.items()}
    if isinstance(obj, list): return [to_py(v) for v in obj]
    return obj

def save_results_json(results, out_path, include_summary=True):
    payload = {"samples": results}
    if include_summary and results:
        payload["summary"] = summarize_metrics(results)
    payload = to_py(payload)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
# ------------------------
# 4) 메인 실행 블록
# ------------------------
if __name__ == "__main__":
    MODEL_PATH = "organ_detector_final_coco.pth"
    ORIGINAL_IMAGE = "image.jpg"
    OUTPUT_DIR   = "test_data_final2_1_2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = OrganTracker(MODEL_PATH, device)

    results = evaluate_with_improvements(
        ORIGINAL_IMAGE, tracker,
        output_dir=OUTPUT_DIR, num_samples=20, iou_threshold=0.3
    )
    save_results_json(results, out_path=f"{OUTPUT_DIR}/evaluation_summary.json", include_summary=True)

    metrics = summarize_metrics(results)
    print("[정확도 요약 지표]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


