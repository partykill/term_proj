import cv2
import numpy as np

# 배경 이미지 크기 조정
background_image = cv2.imread('background.png')
border_size = 25
background_image_resized = np.zeros((1080, 1920, 3), dtype=np.uint8)
background_image_resized[border_size:-border_size, border_size:-border_size] = cv2.resize(background_image, (1920-2*border_size, 1080-2*border_size))

# 웹캠 열기 및 설정
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# 얼굴 감지를 위한 Haar Cascade 분류기 로딩
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 스티커 이미지 로드
sticker_image = cv2.imread('01.png', cv2.IMREAD_UNCHANGED)

# 스티커 크기 설정
sticker_width, sticker_height = 100, 100  # 적절한 크기로 수정하세요

# 이미지 카운트 및 촬영된 이미지 저장 리스트
image_count = 0
captured_images = []

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # 스티커 이미지 크기 조정
            resized_overlay = cv2.resize(sticker_image[:, :, :3], (w, h))

            # 스티커 위치 설정
            roi = frame[y:y+h, x:x+w]
            alpha_channel = sticker_image[:, :, 3] / 255.0
            alpha_channel_resized = cv2.resize(alpha_channel, (w, h))[:, :, None]
            frame[y:y+h, x:x+w, :] = (1 - alpha_channel_resized) * roi + alpha_channel_resized * resized_overlay

        # 배경 이미지와 현재 프레임을 합치기
        combined_image = cv2.addWeighted(frame, 0.5, background_image_resized, 0.5, 0)

        # 화면에 이미지 표시
        cv2.imshow('Video', combined_image)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            image_count += 1
            image_filename = f'captured_image_{image_count}.png'
            cv2.imwrite(image_filename, combined_image)
            print(f'Image captured and saved as {image_filename}')

            captured_image = cv2.imread(image_filename)
            captured_images.append(captured_image)

            if image_count == 3:
                # 이미지를 3번 촬영하면 루프를 종료
                break

    else:
        break

cap.release()

# 이미지 합치기 및 저장
if captured_images:
    merged_image = np.vstack(captured_images)  # 수직으로 이미지 합치기
    cv2.imwrite('merged_image.png', merged_image)
    print("Merged image saved as merged_image_vertical.png")

else:
    print("No images captured.")
