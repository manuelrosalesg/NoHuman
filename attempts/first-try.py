import cv2
import mss
import numpy as np
import time

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Use mss to capture the screen
with mss.mss() as sct:
    monitor = sct.monitors[1]  # Select the first monitor to capture the entire screen

    while True:
        start_time = time.time()  # Record the time when the frame starts

        # Capture a screenshot of the screen
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)  # Convert the screenshot to a numpy array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale for face detection

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

        # Loop through detected faces and draw black rectangles around them
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Fill the rectangle with black color

        # Display the image with black-filled rectangles in a window
        cv2.imshow("Screen Capture - Face Detection", img)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Limit the frame rate to 30 FPS
        elapsed_time = time.time() - start_time  # Calculate how long the frame took to process
        time.sleep(max(0, 1/30 - elapsed_time))  # Sleep to maintain ~30 FPS

# Close all OpenCV windows when the loop ends
cv2.destroyAllWindows()