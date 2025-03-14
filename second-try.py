import sys
import cv2
import mss
import numpy as np
import time
from PyQt5 import QtWidgets, QtGui, QtCore

class OverlayWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Set window attributes for an overlay effect
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # Make the background transparent
        self.showFullScreen()  # Display the widget in full screen
        
        # Load the Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Set up a timer to refresh the overlay at 60 FPS (16.67 ms per frame)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~16.67 ms per frame for 60 FPS
        
        # Initialize the screen capture tool (mss)
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Capture the second monitor (index 1)
        
        # Store detected faces with their timestamp
        self.detected_faces = []  # Will store tuples of (x, y, w, h, timestamp)

    def get_screen_image(self):
        # Capture a screenshot of the monitor
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)  # Convert the screenshot to a NumPy array
        return img

    def detect_faces(self, img):
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def updateFaces(self):
        # Get current screen image and detect faces
        screen_img = self.get_screen_image()
        current_faces = self.detect_faces(screen_img)
        
        current_time = time.time()
        
        # Add newly detected faces to our list with current timestamp
        for (x, y, w, h) in current_faces:
            # Check if this face overlaps significantly with any existing face
            is_new_face = True
            for i, (ex, ey, ew, eh, _) in enumerate(self.detected_faces):
                # Calculate overlap - simple check if centers are close
                if abs(x + w/2 - (ex + ew/2)) < w/2 and abs(y + h/2 - (ey + eh/2)) < h/2:
                    # Update position of existing face and reset timestamp
                    self.detected_faces[i] = (x, y, w, h, current_time)
                    is_new_face = False
                    break
            
            if is_new_face:
                self.detected_faces.append((x, y, w, h, current_time))
        
        # Remove faces that have been displayed for more than 3 seconds
        self.detected_faces = [face for face in self.detected_faces 
                              if current_time - face[4] <= 3.0]

    def paintEvent(self, event):
        # Update the list of faces to display
        self.updateFaces()
        
        # Create a QPainter object to draw on the widget
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)  # Enable antialiasing for smoother drawing
        
        # MAKE BLACK BOX
        painter.setBrush(QtGui.QColor(0, 0, 0))  # Set the brush color to black
        painter.setPen(QtCore.Qt.NoPen)  # Disable the pen (no outline)
        
        # Draw rectangles for all faces in our list
        for (x, y, w, h, timestamp) in self.detected_faces:
            painter.drawRect(x, y, w, h)
        
        # End the painting process
        painter.end()

if __name__ == "__main__":
    # Create the application and the overlay widget
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWidget()
    # Run the application event loop
    sys.exit(app.exec_())