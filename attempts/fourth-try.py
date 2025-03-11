import sys
import cv2
import mss
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import os
import urllib.request

class OverlayWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Set window attributes for overlay effect
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.showFullScreen()
        
        # Initialize Ultra-Light-RFB face detector
        self.initialize_ultra_light_face_detector()
        
        # Set up a timer for refreshing the overlay at 60 FPS
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~16.67 ms per frame for 60 FPS
        
        # Initialize screen capture
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Capture the second monitor (index 1)
        
        # Set color for detection boxes
        self.box_color = QtGui.QColor(255, 0, 0, 128)  # Semi-transparent red
        
    def download_model_if_needed(self, model_url, model_path):
        """Download model if it doesn't exist locally."""
        if not os.path.exists(model_path):
            print(f"Downloading model to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(model_url, model_path)
            print("Download complete")
        return model_path
        
    def initialize_ultra_light_face_detector(self):
        """Initialize the Ultra-Light-RFB face detector."""
        # Define model paths and URLs
        model_folder = os.path.join(os.path.expanduser("~"), ".face_detection_models")
        model_path = os.path.join(model_folder, "RFB-320.onnx")
        model_url = "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/RFB-320.onnx"
        
        # Download model if needed
        try:
            model_path = self.download_model_if_needed(model_url, model_path)
            # Load Ultra-Light-RFB model using OpenCV's DNN module
            self.face_detector = cv2.dnn.readNetFromONNX(model_path)
            self.model_loaded = True
            print("Ultra-Light-RFB model loaded successfully")
        except Exception as e:
            print(f"Error loading Ultra-Light-RFB model: {e}")
            # Fallback to Haar Cascade if Ultra-Light-RFB fails
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.model_loaded = False
            print("Falling back to Haar Cascade face detector")
    
    def get_screen_image(self):
        """Capture a screenshot of the monitor."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        return img
    
    def detect_faces_ultra_light(self, img):
        """Detect faces using Ultra-Light-RFB model."""
        height, width, _ = img.shape
        
        # Prepare input for the model
        blob = cv2.dnn.blobFromImage(
            img, 
            scalefactor=1/255.0, 
            size=(320, 240), 
            mean=(104, 117, 123), 
            swapRB=True, 
            crop=False
        )
        
        # Set input and perform inference
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        # Filter and process detections
        face_boxes = []
        conf_threshold = 0.7
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > conf_threshold:
                # Get coordinates (normalized)
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Convert to (x, y, w, h) format
                face_boxes.append((x1, y1, x2 - x1, y2 - y1))
                
        return face_boxes
    
    def detect_faces_haar(self, img):
        """Fallback face detection using Haar Cascade."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_faces(self, img):
        """Detect faces using the appropriate model."""
        if self.model_loaded:
            return self.detect_faces_ultra_light(img)
        else:
            return self.detect_faces_haar(img)
    
    def paintEvent(self, event):
        """Draw face detection overlay."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Capture screen and detect faces
        screen_img = self.get_screen_image()
        faces = self.detect_faces(screen_img)
        
        # MAKE BLACK BOX
        painter.setBrush(QtGui.QColor(0, 0, 0))  # Set the brush color to black
        painter.setPen(QtCore.Qt.NoPen)  # Disable the pen (no outline)

        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            painter.drawRect(x, y, w, h)
        
        painter.end()

    def keyPressEvent(self, event):
        """Handle key press events."""
        # Press Escape to exit
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayWidget()
    sys.exit(app.exec_())