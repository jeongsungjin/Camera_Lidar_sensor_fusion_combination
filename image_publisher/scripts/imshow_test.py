import cv2
import numpy as np

def test_imshow():
    # 테스트 이미지 생성
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, 'Test Image', (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 이미지 창 표시
    cv2.imshow('Test Image', image)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_imshow()
