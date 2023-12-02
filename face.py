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

            # Resize the overlay image to fit the detected face
            resized_overlay = cv2.resize(overlay_image[:, :, :3], (2 * w, 2 * h))

            # Set the region of interest (ROI) to the resized overlay image
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(resized_overlay, (w, h))

            # Expand alpha_channel to match the number of channels in roi
            alpha_channel = cv2.resize(alpha_channel, (w, h))[:, :, None]

            # Combine the frame and resized overlay using the alpha channel
            frame[y:y+h, x:x+w, :] = (1 - alpha_channel) * frame[y:y+h, x:x+w, :] + \
                                      alpha_channel * roi

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

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
