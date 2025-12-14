import cv2
from ultralytics import YOLO
import json
import requests
import datetime
import os

# 1. 인원수를 혼잡도 레벨(1, 2, 3)로 변환하는 함수 추가
def convert_count_to_crowd_level(person_count):
    """
    감지된 인원수에 따라 혼잡도 레벨(1:낮음, 2:중간, 3:높음)을 반환합니다.
    """
    if person_count <= 3:
        return 1  # 낮음
    elif person_count <= 10:
        return 2  # 중간
    else:
        return 3  # 높음

def detect_persons_and_send_to_server(image_path, api_url, stop_id):
    """
    주어진 이미지 파일에서 사람을 감지하고, 결과를 Spring Boot 서버로 전송합니다.
    """
    # ⚠️ yolo_test.py와 같은 폴더에 있어야 합니다.
    # YOLOv8 모델 로드
    model = YOLO('yolov8n.pt') 

    if not os.path.exists(image_path):
        print(f"오류: 이미지를 찾을 수 없습니다 - {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지 파일을 로드할 수 없습니다 - {image_path}")
        return

    # 이미지에서 객체 감지 수행
    results = model(img)

    person_count = 0
    for r in results:
        # 'person' 클래스 ID는 0입니다.
        person_count += len(r.boxes[r.boxes.cls == 0]) 
    
    # --- 결과 출력 ---
    print("\n" + "="*50)
    print(f"✅ 이미지 '{image_path}' 감지 완료!")
    print(f"   -> 감지된 총 인원수: {person_count}명")
    
    # 1. 인원수를 혼잡도 레벨로 변환
    crowd_level = convert_count_to_crowd_level(person_count)
    print(f"   -> 변환된 혼잡도 레벨: {crowd_level} (DB 저장값)")
    print("="*50)

    # 2. 서버로 보낼 JSON 데이터 생성 (Spring StopController 형식)
    data_to_send = {
        "crowd": crowd_level
    }

    # 3. 서버 통신 (API_URL은 호출부에서 로컬 IP로 설정됩니다.)
    print(f"서버 '{api_url}'로 데이터 전송 시도: {json.dumps(data_to_send, indent=2)}")
    try:
        # 실제 서버 통신: StopController의 POST /api/stops/{stopId}/crowd 호출
        response = requests.post(api_url, json=data_to_send)
        response.raise_for_status() 
        print(f"데이터 전송 성공! 서버 응답: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"오류: 서버로 데이터 전송 실패 - {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"서버 응답 내용: {e.response.text}")
        
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")


if __name__ == "__main__":
    # --- 설정 값 (로컬 테스트용) ---
    TEST_IMAGE_PATH = "test_image.jpg" 
    
    # DB에 혼잡도를 업데이트할 정류장 ID 지정 (DB에 정의된 값: 'baekseok', 'terminal' 등)
    STOP_ID = "baekseok" 

    # 서버 IP 주소를 로컬 호스트(127.0.0.1)로 명확하게 설정
    SERVER_IP = "13.211.215.48"
    
    # API 주소: http://127.0.0.1:8080/api/stops/{stopId}/crowd
    # ⚠️ f-string을 사용하여 중괄호 {} 없이 URL을 구성
    SPRING_BOOT_API_URL = f"http://{SERVER_IP}:8080/api/stops/{STOP_ID}/crowd"
    # --- 설정 값 끝 ---

    detect_persons_and_send_to_server(TEST_IMAGE_PATH, SPRING_BOOT_API_URL, STOP_ID)