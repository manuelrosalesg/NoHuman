import sys
import cv2
import mss
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore

class OverlayWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Set window attributes for overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.showFullScreen()

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.4
        )

        # Timer for 60 FPS refresh
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)  # 60 FPS

        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

    def get_screen_image(self):
        screenshot = self.sct.grab(self.monitor)
        return np.array(screenshot)

    def detect_faces(self, img):
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_img)
        
        faces = []
        if results.detections:
            height, width, _ = img.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * width)
                y = int(bboxC.ymin * height)
                w = int(bboxC.width * width)
                h = int(bboxC.height * height)
                faces.append((x, y, w, h))
        
        return faces, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def convert_cv_to_qimage(self, cv_img):
        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        if channels == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(cv_img.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Clear previous frame
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

        screen_img = self.get_screen_image()
        faces, gray_img = self.detect_faces(screen_img)

        for (x, y, w, h) in faces:
            # Extract face region and apply binary threshold
            face_region = gray_img[y:y+h, x:x+w]
            _, binary_face = cv2.threshold(face_region, 127, 255, cv2.THRESH_BINARY)
            
            # Convert to 3-channel image for QImage compatibility
            try:
                binary_face_3c = cv2.cvtColor(binary_face, cv2.COLOR_GRAY2BGR)
            except cv2.error:
                pass
            else:
                # Convert to QImage and draw
                qimg = self.convert_cv_to_qimage(binary_face_3c)
                painter.drawImage(QtCore.QPoint(x, y), qimg)

                # Draw red border
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
                painter.drawRect(x, y, w, h)

        painter.end()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWidget()
    sys.exit(app.exec_())