from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid

app = FastAPI(title="People Counter API", description="API to count people in images")

# === CrowdHuman / 커스텀 데이터셋 설정 영역 ==========================
# CrowdHuman을 YOLO 형식으로 변환해서 학습했다는 전제:
#   예) 단일 클래스: ["person"]
#   예) 다중 클래스: ["person_full", "person_visible", "head"]
#
# 실제 학습 시 사용한 클래스 이름과 동일하게 맞춰주세요.
TARGET_CLASS_NAMES = {
    "person",          # 단일 클래스인 경우
    "person_full",     # 전체 인체
    "person_visible",  # 보이는 부분
    "head"             # 머리 박스
}

# 너무 낮은 confidence 박스는 무시
MIN_CONFIDENCE = 0.3
# ============================================================

# 결과 이미지 저장 폴더
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# /outputs 경로로 정적 파일 제공 (이미지 다운로드/보기용)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# YOLO 모델 (처음 1번만 로드)
model = None

def load_model():
    """Load YOLO model for CrowdHuman-based person detection"""
    global model
    if model is None:
        # CrowdHuman으로 학습한 커스텀 가중치
        # (이미 Colab/서버에 best.pt를 업로드 했다고 가정)
        model = YOLO("best.pt")
    return model

@app.get("/")
async def root():
    return {"message": "People Counter API - Use POST /count to upload an image"}

@app.post("/count")
async def count_people(
    request: Request,
    file: UploadFile = File(...)
):
    """
    이미지를 업로드 받아서:
    - 사람 수를 세고
    - 감지된 사람에 바운딩 박스를 그린 이미지를 저장한 뒤
    - 그 이미지 URL을 반환
    """
    # 1. 파일 타입 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # 2. 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")  # RGB로 강제 변환

        # 3. PIL → numpy → BGR(OpenCV용)
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # 4. YOLO 모델 로드 & 추론
        yolo_model = load_model()
        results = yolo_model(image_bgr)[0]  # 한 장이므로 [0] 사용

        # 모델에 저장된 클래스 이름 딕셔너리 가져오기
        # 예: {0: 'person_full', 1: 'person_visible', 2: 'head'}
        names = yolo_model.model.names if hasattr(yolo_model, "model") else yolo_model.names

        # 5. 사람 수 세기 및 바운딩 박스 그리기
        person_count = 0
        annotated_img = image_bgr.copy()

        boxes = results.boxes
        if boxes is not None:
            for box in boxes:
                # confidence 필터
                conf = float(box.conf[0]) if box.conf is not None else 1.0
                if conf < MIN_CONFIDENCE:
                    continue

                # 클래스 id → 클래스 이름
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id])

                # CrowdHuman 커스텀 클래스 중 사람으로 취급할 클래스만 카운트
                if cls_name not in TARGET_CLASS_NAMES:
                    continue

                person_count += 1

                # 박스 좌표 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 바운딩 박스 (초록색, 두께 2)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 라벨 텍스트: 클래스 이름 + confidence (선택)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # 6. 결과 이미지 저장
        output_filename = f"{uuid.uuid4().hex}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        success = cv2.imwrite(output_path, annotated_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save result image")

        # 7. 절대 URL 만들기 (예: https://xxxx.ngrok-free.app/outputs/xxxx.png)
        base_url = str(request.base_url).rstrip("/")
        result_image_url = f"{base_url}/outputs/{output_filename}"

        # 8. JSON 응답
        return JSONResponse(content={
            "count": person_count,
            "filename": file.filename,
            "result_image_url": result_image_url
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
