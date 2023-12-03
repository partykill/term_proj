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

        # Loop over all detected faces
        for (x, y, w, h) in faces:
            # Calculate the position for the image next to the face
            x_offset = x + w
            y_offset = y

            # Resize the overlay image to fit the detected face
            resized_overlay = cv2.resize(overlay_image[:, :, :3], (w, h))

            # Display the overlay image next to the face
            overlay_height, overlay_width = resized_overlay.shape[:2]
            frame[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = resized_overlay

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




