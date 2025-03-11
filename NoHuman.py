import sys
import cv2
import mss
import numpy as np
import time
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
        self.timer.start(1)  # ~16.67 ms for 60 FPS
        
        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        
        # Store detected people with their timestamp
        self.detected_people = []  # Will store tuples of (x, y, w, h, timestamp)

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
        return people
    
    def updatePeople(self):
        # Get current screen image and detect people
        screen_img = self.get_screen_image()
        current_people = self.detect_people(screen_img)
        
        current_time = time.time()
        
        # Add newly detected people to our list with current timestamp
        for (x, y, w, h) in current_people:
            # Check if this person overlaps significantly with any existing person
            is_new_person = True
            for i, (ex, ey, ew, eh, timestamp) in enumerate(self.detected_people):
                # Calculate overlap - simple check if centers are close
                if abs(x + w/2 - (ex + ew/2)) < w/2 and abs(y + h/2 - (ey + eh/2)) < h/2:
                    # Update position of existing person and reset timestamp
                    self.detected_people[i] = (x, y, w, h, current_time)
                    is_new_person = False
                    break
            
            if is_new_person:
                self.detected_people.append((x, y, w, h, current_time))
        
        # Remove people that have been displayed for more than 3 seconds
        self.detected_people = [person for person in self.detected_people 
                               if current_time - person[4] <= 2.5]

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Clear previous frame
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        
        # Update the people list
        self.updatePeople()
        
        # Draw all people in our list
        for (x, y, w, h, timestamp) in self.detected_people:
            # Draw black box over detected person
            painter.setBrush(QtGui.QBrush(QtCore.Qt.black))  # Black color for the box
            painter.setPen(QtCore.Qt.NoPen)  # No outline
            painter.drawRect(x, y, w, h)  # Draw rectangle
        
        painter.end()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWidget()
    sys.exit(app.exec_())