import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

sticker_image = cv2.imread('03.png', cv2.IMREAD_UNCHANGED)

# Check if the sticker image has an alpha channel
if sticker_image.shape[2] == 4:
    # If the image has an alpha channel, use the alpha channel for transparency
    alpha_channel = sticker_image[:, :, 3] / 255.0
    sticker = sticker_image[:, :, :3]  # Extract RGB channels
else:
    # If the image does not have an alpha channel, use it as is
    alpha_channel = None
    sticker = sticker_image

image_count = 0
captured_images = []

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(300, 300))

        for (x, y, w, h) in faces:
            if alpha_channel is not None:
                # Resize the sticker and overlay it on the face, 10 pixels above the face
                sticker_height = int(h * 0.5)
                sticker_width = int((sticker_height / sticker.shape[0]) * sticker.shape[1])
                resized_overlay = cv2.resize(sticker, (sticker_width, sticker_height))
                sticker_y = max(0, y - 200)
                roi = frame[sticker_y:sticker_y+sticker_height, x:x+sticker_width]

                # Expand alpha_channel to match the number of channels in roi
                alpha_channel_roi = cv2.resize(alpha_channel, (roi.shape[1], roi.shape[0]))[:, :, None]

                # Combine the frame and resized overlay using the alpha channel
                frame[sticker_y:sticker_y+sticker_height, x:x+sticker_width, :] = \
                    (1 - alpha_channel_roi) * frame[sticker_y:sticker_y+sticker_height, x:x+sticker_width, :] + \
                    alpha_channel_roi * resized_overlay
            else:
                # Resize the sticker and overlay it on the face, 10 pixels above the face
                sticker_height = int(h * 0.5)
                sticker_width = int((sticker_height / sticker.shape[0]) * sticker.shape[1])
                sticker = cv2.resize(sticker, (sticker_width, sticker_height))
                sticker_y = max(0, y - 200)
                frame[sticker_y:sticker_y+sticker_height, x:x+sticker_width] = sticker

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        frame = cv2.flip(frame, 1)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            image_count += 1
            image_filename = f'captured_image_{image_count}.png'
            cv2.imwrite(image_filename, frame)
            print(f'Image captured and saved as {image_filename}')

            captured_image = cv2.imread(image_filename)
            captured_images.append(captured_image)

            # If 3 images are captured, exit the loop
            if image_count == 3:
                break

    else:
        break

cap.release()

# Merge captured images vertically and save the result
if captured_images:
    merged_image = np.vstack(captured_images)  # 수직으로 이미지 합치기
    cv2.imwrite('merged_image.png', merged_image)
    print("Vertical Merged image saved as merged_image_vertical.png")

else:
    print("No images captured.")
