import cv2
import numpy as np

# Open the webcam with resolution 1920x1080
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # Set height

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load overlay images
overlay_images = [
    cv2.imread('01.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('02.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('03.png', cv2.IMREAD_UNCHANGED),  # For filter 3
]

# Variables to keep track of captured image count
image_counts = [0, 0, 0]

# Captured images
captured_images = [[], [], []]

filter_idx = 0  # Index of the current filter (1, 2, 3)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(300, 300))

        for (x, y, w, h) in faces:
            alpha_channel = overlay_images[filter_idx][:, :, 3] / 255.0

            if filter_idx == 0:  # Filter 1
                expanded_w = w + 200
                expanded_h = h + 200
                resized_overlay = cv2.resize(overlay_images[filter_idx][:, :, :3], (expanded_w, expanded_h))
                roi_x = max(x - 100, 0)
                roi_y = max(y - 100, 0)
                roi = frame[roi_y:roi_y + expanded_h, roi_x:roi_x + expanded_w]
                roi = cv2.resize(resized_overlay, (roi.shape[1], roi.shape[0]))
                alpha_channel = cv2.resize(alpha_channel, (roi.shape[1], roi.shape[0]))[:, :, None]
                frame[roi_y:roi_y + expanded_h, roi_x:roi_x + expanded_w, :] = \
                    (1 - alpha_channel) * frame[roi_y:roi_y + expanded_h, roi_x:roi_x + expanded_w, :] + \
                    alpha_channel * roi

            elif filter_idx == 1:  # Filter 2
                x_offset = x + w
                y_offset = y
                alpha_channel = overlay_images[filter_idx][:, :, 3] / 255.0
                resized_overlay = cv2.resize(overlay_images[filter_idx][:, :, :3], (w, h))
                for c in range(0, 3):
                    resized_alpha = cv2.resize(alpha_channel, (w, h))
                    frame[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
                        frame[y_offset:y_offset + h, x_offset:x_offset + w, c] * (1 - resized_alpha) + \
                        resized_overlay[:, :, c] * resized_alpha

            elif filter_idx == 2:  # Filter 3
                sticker_image = overlay_images[filter_idx]
                if sticker_image.shape[2] == 4:
                    alpha_channel = sticker_image[:, :, 3] / 255.0
                    sticker = sticker_image[:, :, :3]
                else:
                    alpha_channel = None
                    sticker = sticker_image

                if alpha_channel is not None:
                    sticker_height = int(h * 0.5)
                    sticker_width = int((sticker_height / sticker.shape[0]) * sticker.shape[1])
                    resized_overlay = cv2.resize(sticker, (sticker_width, sticker_height))
                    sticker_y = max(0, y - 200)
                    roi = frame[sticker_y:sticker_y + sticker_height, x:x + sticker_width]
                    alpha_channel_roi = cv2.resize(alpha_channel, (roi.shape[1], roi.shape[0]))[:, :, None]
                    frame[sticker_y:sticker_y + sticker_height, x:x + sticker_width, :] = \
                        (1 - alpha_channel_roi) * frame[sticker_y:sticker_y + sticker_height, x:x + sticker_width, :] + \
                        alpha_channel_roi * resized_overlay
                else:
                    sticker_height = int(h * 0.5)
                    sticker_width = int((sticker_height / sticker.shape[0]) * sticker.shape[1])
                    sticker = cv2.resize(sticker, (sticker_width, sticker_height))
                    sticker_y = max(0, y - 200)
                    frame[sticker_y:sticker_y + sticker_height, x:x + sticker_width] = sticker

            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame = cv2.flip(frame, 1)
        cv2.imshow('Video', frame)

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            image_counts[filter_idx] += 1
            image_filename = f'captured_image_{image_counts[filter_idx]}_filter_{filter_idx + 1}.png'
            cv2.imwrite(image_filename, frame)
            print(f'Image captured and saved as {image_filename}')

            captured_image = cv2.imread(image_filename)
            captured_images[filter_idx].append(captured_image)

            filter_idx += 1
            if filter_idx == 3:
                filter_idx = 0

    else:
        break

cap.release()

# Merge captured images vertically and save the result
merged_images = [np.vstack(images) for images in captured_images if images]
if merged_images:
    merged_image = np.vstack(merged_images)  # 수직으로 이미지 합치기
    cv2.imwrite('merged_image.png', merged_image)
    print("Vertical Merged images saved as merged_image.png")
else:
    print("No images captured.")
cv2.destroyAllWindows()
