import cv2

# 웹캠을 1920x1080 해상도로 엽니다
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # 가로 크기 설정
cap.set(4, 1080)  # 세로 크기 설정

# 얼굴을 감지하는 Haar Cascade 분류기를 로드합니다
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 알파 채널을 포함한 오버레이 이미지를 로드합니다
overlay_image = cv2.imread('03.png', cv2.IMREAD_UNCHANGED)

# 캡쳐된 이미지 수를 기록하는 변수
image_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(300, 300))

        # 감지된 모든 얼굴에 대해 반복합니다
        for (x, y, w, h) in faces:
            # 얼굴 옆에 이미지를 표시할 위치를 계산합니다
            x_offset = x + w
            y_offset = y

            # 오버레이 이미지에서 알파 채널을 추출합니다
            alpha_channel = overlay_image[:, :, 3] / 255.0

            # 감지된 얼굴에 오버레이 이미지를 맞추기 위해 크기를 조절합니다
            resized_overlay = cv2.resize(overlay_image[:, :, :3], (w, h))

            # 알파 블렌딩을 적용하여 얼굴에 이미지를 오버레이합니다
            for c in range(0, 3):
                resized_alpha = cv2.resize(alpha_channel, (w, h))
                frame[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
                frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1 - resized_alpha) + \
                resized_overlay[:, :, c] * resized_alpha


            # 얼굴 주위에 사각형 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 프레임을 좌우 반전시킵니다
        frame = cv2.flip(frame, 1)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(25) & 0xFF

        # 'q'를 눌러 루프를 종료합니다
        if key == ord('q'):
            break
        # 'c'를 눌러 이미지를 캡처하고 저장합니다
        elif key == ord('c'):
            image_count += 1
            image_filename = f'captured_image_{image_count}.png'
            cv2.imwrite(image_filename, frame)
            print(f'이미지가 캡처되어 {image_filename}로 저장되었습니다')

    else:
        break

cap.release()
cv2.destroyAllWindows()
