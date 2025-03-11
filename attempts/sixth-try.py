import sys
import cv2
import mss
import numpy as np
from ultralytics import YOLO
from PyQt5 import QtWidgets, QtGui, QtCore

class OverlayWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Set window attributes for overlay
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.showFullScreen()

        # Load YOLOv8 Pose model
        self.model = YOLO("yolov8n-pose.pt")  # Smallest model for speed

        # Timer for 60 FPS refresh
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)  # 60 FPS

        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

    def get_screen_image(self):
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)  # Convert MSS screenshot to NumPy array
        if img.shape[-1] == 4:  # If the image has 4 channels (RGBA)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR (3 channels)
        return img


    def detect_people(self, img):
        """Detect multiple people using YOLOv8 Pose."""
        results = self.model(img)  # Run YOLO on image

        people = []
        for result in results:
            for box in result.boxes.xyxy:  # Bounding boxes
                x1, y1, x2, y2 = map(int, box[:4])
                people.append((x1, y1, x2 - x1, y2 - y1))  # (x, y, width, height)

        return people, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def convert_cv_to_qimage(self, cv_img):
        height, width = cv_img.shape[:2]
        bytes_per_line = width
        return QtGui.QImage(cv_img.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Clear previous frame
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

        screen_img = self.get_screen_image()
        people, gray_img = self.detect_people(screen_img)

        for (x, y, w, h) in people:
            # Extract person region and apply binary threshold
            person_region = gray_img[y:y+h, x:x+w]
            _, binary_person = cv2.threshold(person_region, 127, 255, cv2.THRESH_BINARY)

            # Convert to QImage and draw
            qimg = self.convert_cv_to_qimage(binary_person)
            painter.drawImage(QtCore.QPoint(x, y), qimg)

            # Draw white border
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
