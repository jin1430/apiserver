from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, os, uuid, io, math
import uvicorn
import requests
import oracledb

from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="People Counter (A: Crowd GAP Improved)")

# -------------------------------------------------------------
# 1. CORS 설정 (웹 대시보드 접속 허용)
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# 모델 로드
model = YOLO("yolo11n.pt")

# 군중용 추론 파라미터
PRED_CONF = 0.15
PRED_IOU = 0.60
PRED_CLASSES = [0]
PRED_AGNOSTIC_NMS = False

# -------------------- DB 저장 함수 --------------------
def save_to_db(stop_id, level_str):
    try:
        # DB 연결 (ID/PW 확인 필수)
        conn = oracledb.connect(
            user="BUS_USER",   # ⚠️ 테이블을 만든 계정이 맞는지 확인하세요
            password="BusPass1234!",
            dsn="localhost:1521/xe"
        )
        cursor = conn.cursor()

        sql = "INSERT INTO bus_congestion (stop_id, congestion_level, created_at) VALUES (:1, :2, SYSDATE)"
        cursor.execute(sql, [stop_id, level_str])
        conn.commit()
        print(f"✅ Oracle DB 저장 성공: {stop_id}, {level_str}")

    except Exception as e:
        print(f"❌ Oracle DB 저장 실패: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# -------------------- utilities --------------------
def extract_person_boxes(results):
    boxes = []
    if results.boxes is None:
        return boxes
    for b in results.boxes:
        if int(b.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0]) if b.conf is not None else 0.0
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})
    return boxes

def sort_lr(boxes):
    return sorted(boxes, key=lambda b: (b["x1"] + b["x2"]) / 2)

def centers(boxes):
    return [((b["x1"] + b["x2"]) // 2, (b["y1"] + b["y2"]) // 2) for b in boxes]

def distances_2d(sorted_boxes):
    c = centers(sorted_boxes)
    d = []
    for i in range(len(c) - 1):
        dx = c[i+1][0] - c[i][0]
        dy = c[i+1][1] - c[i][1]
        d.append(float(math.sqrt(dx*dx + dy*dy)))
    return c, d

def visualize_dist(image, boxes):
    out = image.copy()
    s = sort_lr(boxes)
    c, d = distances_2d(s)
    for i, (x, y) in enumerate(c):
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(out, str(i), (x + 6, y - 6), 0, 0.6, (0, 0, 255), 2)
    for i, dist in enumerate(d):
        (x1, y1), (x2, y2) = c[i], c[i+1]
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(out, f"{int(dist)}px", ((x1 + x2)//2, (y1 + y2)//2),
                    0, 0.7, (255, 0, 0), 2)
    return out, d

# -------------------- robust gap finder --------------------
def trimmed_median_threshold(dists, gap_multiplier=2.2, min_gap_px=60):
    if not dists:
        return 0.0
    arr = np.array(dists, dtype=np.float32)
    arr.sort()
    n = len(arr)
    if n >= 10:
        trim = int(n * 0.15)
        arr2 = arr[trim:n-trim] if (n - 2*trim) >= 3 else arr
    else:
        arr2 = arr
    med = float(np.median(arr2))
    thr = max(med * gap_multiplier, float(min_gap_px))
    return thr

def gap_regions_crowd(boxes, img_w, gap_multiplier=2.2, min_gap_px=60, min_region_w=80, margin_ratio=0.8, margin_px=120):
    s = sort_lr(boxes)
    c, d = distances_2d(s)
    if not d:
        return [], 0.0, d
    thr = trimmed_median_threshold(d, gap_multiplier=gap_multiplier, min_gap_px=min_gap_px)
    gaps = []
    for i, dist in enumerate(d):
        if dist > thr:
            x_left = s[i]["x2"]
            x_right = s[i+1]["x1"]
            gap_w = max(0, x_right - x_left)
            margin = int(gap_w * margin_ratio) + margin_px
            x1 = max(0, x_left - margin)
            x2 = min(img_w, x_right + margin)
            if (x2 - x1) >= min_region_w:
                gaps.append((x1, x2))
    gaps.sort()
    merged = []
    for g in gaps:
        if not merged or g[0] > merged[-1][1]:
            merged.append(list(g))
        else:
            merged[-1][1] = max(merged[-1][1], g[1])
    merged = [tuple(m) for m in merged]
    return merged, thr, d

def draw_gap_image(img, gaps, thr_text=None):
    out = img.copy()
    h, w = out.shape[:2]
    for (x1, x2) in gaps:
        cv2.rectangle(out, (x1, 0), (x2, h-1), (0, 255, 255), 2)
        cv2.putText(out, f"GAP [{x1},{x2}]", (x1+5, 30), 0, 0.7, (0, 255, 255), 2)
    if thr_text is not None:
        cv2.putText(out, thr_text, (10, h-10), 0, 0.8, (0, 255, 255), 2)
    return out

# -------------------- final dedup nms --------------------
def iou(a, b):
    xA = max(a["x1"], b["x1"])
    yA = max(a["y1"], b["y1"])
    xB = min(a["x2"], b["x2"])
    yB = min(a["y2"], b["y2"])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(1, (a["x2"] - a["x1"])) * max(1, (a["y2"] - a["y1"]))
    areaB = max(1, (b["x2"] - b["x1"])) * max(1, (b["y2"] - b["y1"]))
    return inter / (areaA + areaB - inter + 1e-9)

def dedup_nms(boxes, iou_thr=0.70):
    boxes = sorted(boxes, key=lambda b: b.get("conf", 0.0), reverse=True)
    keep = []
    for b in boxes:
        if all(iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep

@app.get("/")
async def root():
    return {"message": "A: Crowd GAP Improved - POST /count"}

# -------------------------------------------------------------
# 2. 웹 대시보드용 조회 API
# -------------------------------------------------------------
@app.get("/api/stops/{stop_id}")
async def get_congestion(stop_id: str):
    try:
        conn = oracledb.connect(
            user="BUS_USER",
            password="BusPass1234!",
            dsn="localhost:1521/xe"
        )
        cursor = conn.cursor()

        # 가장 최근 데이터 1개 조회
        sql = """
            SELECT congestion_level 
            FROM bus_congestion 
            WHERE stop_id = :1 
            ORDER BY created_at DESC 
            FETCH FIRST 1 ROWS ONLY
        """
        cursor.execute(sql, [stop_id])
        row = cursor.fetchone()

        if row is None:
            return {"crowd": 0}

        level_str = row[0]
        crowd_score = 1

        if level_str == 'High': crowd_score = 3
        elif level_str == 'Normal': crowd_score = 2
        elif level_str == 'Low': crowd_score = 1

        return {"crowd": crowd_score}

    except Exception as e:
        print(f"DB Error: {e}")
        return {"crowd": 0}
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# -------------------------------------------------------------
# 3. 데이터 수신 및 처리 API
# -------------------------------------------------------------
@app.post("/count")
async def count(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1차 탐지
    r1 = model.predict(
        img, conf=PRED_CONF, iou=PRED_IOU, classes=PRED_CLASSES, agnostic_nms=PRED_AGNOSTIC_NMS
    )[0]
    boxes = extract_person_boxes(r1)

    # 거리 디버그
    debug_img, dists = visualize_dist(img, boxes)
    dbg_name = f"debug_{uuid.uuid4().hex}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, dbg_name), debug_img)

    # GAP 찾기
    gaps, threshold, _ = gap_regions_crowd(
        boxes, img_w=img.shape[1],
        gap_multiplier=2.2, min_gap_px=60,
        min_region_w=120, margin_ratio=0.8, margin_px=140
    )

    # GAP 시각화
    gap_img = draw_gap_image(img, gaps, thr_text=f"thr={threshold:.1f}px")
    gap_name = f"gap_{uuid.uuid4().hex}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, gap_name), gap_img)

    # 2차 재탐지
    for x1, x2 in gaps:
        crop = img[:, x1:x2]
        if crop.shape[1] < 40:
            continue
        r2 = model.predict(
            crop, conf=PRED_CONF, iou=PRED_IOU, classes=PRED_CLASSES, agnostic_nms=PRED_AGNOSTIC_NMS
        )[0]
        for b in extract_person_boxes(r2):
            b["x1"] += x1
            b["x2"] += x1
            boxes.append(b)

    # 최종 중복 제거
    boxes = dedup_nms(boxes, iou_thr=0.70)

    # -------------------------------------------------------------
    # 혼잡도 계산 및 저장
    # -------------------------------------------------------------
    person_count = len(boxes)
    crowd_level = 1
    level_str = "Low"

    if person_count >= 10:
        crowd_level = 3
        level_str = "High"
    elif person_count >= 5:
        crowd_level = 2
        level_str = "Normal"
    else:
        crowd_level = 1
        level_str = "Low"

    # 1. Oracle DB에 저장
    save_to_db("baekseok", level_str)

    # 2. Java 서버로 전송
    try:
        java_url = "http://localhost:8080/api/stops/baekseok/crowd"
        payload = {"crowd": crowd_level}
        requests.post(java_url, json=payload, timeout=2)
        print(f"✅ Java 서버 전송 성공: 사람수={person_count}, 혼잡도={crowd_level}")
    except Exception as e:
        print(f"❌ Java 서버 전송 실패: {e}")

    # 이미지 저장 및 반환
    out = img.copy()
    for b in boxes:
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)

    name = f"{uuid.uuid4().hex}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), out)

    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "count": len(boxes),
        "predict": {"conf": PRED_CONF, "iou": PRED_IOU, "agnostic_nms": PRED_AGNOSTIC_NMS},
        "gap_threshold_px": float(threshold),
        "result_image_url": f"{base}/outputs/{name}",
        "debug_image_url": f"{base}/outputs/{dbg_name}",
        "gap_image_url": f"{base}/outputs/{gap_name}",
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)