import cv2

# Open the webcam with resolution 1920x1080
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # Set height

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the overlay image
overlay_image = cv2.imread('01.png', cv2.IMREAD_UNCHANGED)

# Variable to keep track of captured image count
image_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.05, 4, minSize=(300, 300))

        for (x, y, w, h) in faces:
            # Extract the alpha channel from the overlay image
            alpha_channel = overlay_image[:, :, 3] / 255.0

            # Resize the overlay image to fit the detected face and expand by 20 pixels in each direction
            expanded_w = w + 200
            expanded_h = h + 200
            resized_overlay = cv2.resize(overlay_image[:, :, :3], (expanded_w, expanded_h))

            # Set the region of interest (ROI) to the resized overlay image
            roi_x = max(x - 100, 0)  # Ensure x coordinate does not go below 0
            roi_y = max(y - 100, 0)  # Ensure y coordinate does not go below 0
            roi = frame[roi_y:roi_y + expanded_h, roi_x:roi_x + expanded_w]
            roi = cv2.resize(resized_overlay, (roi.shape[1], roi.shape[0]))  # Resize to match roi size

            # Expand alpha_channel to match the number of channels in roi
            alpha_channel = cv2.resize(alpha_channel, (roi.shape[1], roi.shape[0]))[:, :, None]

            # Combine the frame and resized overlay using the alpha channel
            frame[roi_y:roi_y + expanded_h, roi_x:roi_x + expanded_w, :] = \
                (1 - alpha_channel) * frame[roi_y:roi_y + expanded_h, roi_x:roi_x + expanded_w, :] + \
                alpha_channel * roi

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Flip the frame horizontally (left-to-right)
        frame = cv2.flip(frame, 1)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(25) & 0xFF

        # Press 'q' to exit the loop
        if key == ord('q'):
            break
        # Press 'c' to capture and save an image
        elif key == ord('c'):
            image_count += 1
            image_filename = f'captured_image_{image_count}.png'
            cv2.imwrite(image_filename, frame)
            print(f'Image captured and saved as {image_filename}')

    else:
        break

cap.release()
cv2.destroyAllWindows()
