import sys
import cv2
import mss
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore

class OverlayWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Set window attributes for overlay effect
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.showFullScreen()
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        
        # Set up a timer for refreshing the overlay at 60 FPS
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)  # ~16.67 ms per frame for 60 FPS
        
        # Initialize screen capture
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Capture the second monitor (index 1)

    def get_screen_image(self):
        """Capture a screenshot of the monitor."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        return img

    def detect_faces(self, img):
        """Detect faces using MediaPipe."""
        # Convert image to RGB (MediaPipe requires RGB format)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_img)

        faces = []
        if results.detections:
            height, width, _ = img.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = (
                    int(bboxC.xmin * width),
                    int(bboxC.ymin * height),
                    int(bboxC.width * width),
                    int(bboxC.height * height)
                )
                faces.append((x, y, w, h))
        
        return faces

    def paintEvent(self, event):
        """Draw face detection overlay."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # MAKE BLACK BOX
        painter.setBrush(QtGui.QColor(0, 0, 0))  # Set the brush color to black
        painter.setPen(QtCore.Qt.NoPen)  # Disable the pen (no outline)

        
        # Capture screen and detect faces
        screen_img = self.get_screen_image()
        faces = self.detect_faces(screen_img)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            painter.drawRect(x, y, w, h)
        
        painter.end()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWidget()
    sys.exit(app.exec_())
