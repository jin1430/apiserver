import cv2
from ultralytics import YOLO
import json
import requests
import datetime
import os

def detect_persons_and_send_to_server(image_path, api_url):
    """
    주어진 이미지 파일에서 사람을 감지하고, 인원수를 서버로 전송합니다.
    """
    try:
        # YOLOv8 모델 로드 (n = nano 모델, 가장 작고 빠름)
        # 이 모델은 처음 실행 시 자동으로 다운로드됩니다.
        model = YOLO('yolov8n.pt')

        # 이미지 로드
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
            # r.boxes는 감지된 모든 객체의 바운딩 박스를 포함
            for box in r.boxes:
                # box.cls는 클래스 ID, box.conf는 신뢰도
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # YOLOv8 COCO 데이터셋에서 'person'의 클래스 ID는 0입니다.
                if model.names[class_id] == 'person' and confidence > 0.5: # 50% 이상 신뢰도만 카운트
                    person_count += 1

        print(f"이미지 '{image_path}'에서 감지된 인원수: {person_count}명")

        # 서버로 보낼 JSON 데이터 생성
        data_to_send = {
            "timestamp": datetime.datetime.now().isoformat(),
            "location": "Test Location A", # 실제 위치로 변경 가능
            "personCount": person_count
        }

        # Spring Boot 서버로 데이터 전송 (POST 요청)
        try:
            print(f"서버 '{api_url}'로 데이터 전송 시도: {json.dumps(data_to_send, indent=2)}")
            response = requests.post(api_url, json=data_to_send)
            response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
            print(f"데이터 전송 성공! 서버 응답: {response.status_code} - {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"오류: 서버로 데이터 전송 실패 - {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"서버 응답 내용: {e.response.text}")

    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    # --- 설정 값 ---
    # 테스트할 이미지 파일 경로 (서버에 미리 업로드해야 합니다!)
    # 예: 'test_image.jpg' 또는 'images/my_test_pic.png'
    TEST_IMAGE_PATH = "test_image.jpg"

    # Spring Boot 서버의 API 엔드포인트 URL (실제 서버 IP와 포트로 변경!)
    # 예: http://13.211.215.48:8080/api/person-count
    # [주의!] {YOUR_SERVER_IP} 부분을 실제 EC2 인스턴스의 퍼블릭 IP로 변경하세요.
    SPRING_BOOT_API_URL = "http://{YOUR_SERVER_IP}:8080/api/person-count"
    # --- 설정 값 끝 ---

    detect_persons_and_send_to_server(TEST_IMAGE_PATH, SPRING_BOOT_API_URL)