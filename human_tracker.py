import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys
import threading
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGroupBox, QGridLayout, QSlider, QCheckBox,
                             QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from collections import deque


class MotionPredictor:
    """Predicts future target positions based on movement history"""

    def __init__(self, history_size=10):
        self.position_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        self.velocity_x = 0
        self.velocity_y = 0
        self.last_update_time = 0
        self.prediction_active = False
        self.prediction_start_time = 0
        self.max_prediction_time = 3.0  # seconds to continue prediction

    def add_position(self, x, y):
        """Add a new position to the history"""
        current_time = time.time()
        self.position_history.append((x, y))
        self.timestamp_history.append(current_time)

        # Update velocity if we have enough history
        if len(self.position_history) >= 2:
            # Calculate velocity from the last few positions
            time_diff = self.timestamp_history[-1] - self.timestamp_history[0]
            if time_diff > 0:
                pos_diff_x = self.position_history[-1][0] - self.position_history[0][0]
                pos_diff_y = self.position_history[-1][1] - self.position_history[0][1]
                self.velocity_x = pos_diff_x / time_diff
                self.velocity_y = pos_diff_y / time_diff

        self.last_update_time = current_time
        self.prediction_active = False  # Reset prediction when we have a real detection

    def start_prediction(self):
        """Start predicting movement when target is lost"""
        if len(self.position_history) < 2:
            return False  # Not enough history to predict

        self.prediction_active = True
        self.prediction_start_time = time.time()
        return True

    def get_predicted_position(self):
        """Get the predicted current position based on last known position and velocity"""
        if not self.prediction_active or len(self.position_history) < 1:
            return 0.5, 0.5  # Default center position

        # Check if prediction has timed out
        if time.time() - self.prediction_start_time > self.max_prediction_time:
            self.prediction_active = False
            return 0.5, 0.5  # Return to center if prediction times out

        # Calculate time since last known position
        time_since_last = time.time() - self.last_update_time

        # Get last known position
        last_x, last_y = self.position_history[-1]

        # Calculate predicted position
        pred_x = last_x + self.velocity_x * time_since_last
        pred_y = last_y + self.velocity_y * time_since_last

        # Clamp predicted position to valid range with some buffer
        pred_x = max(0.1, min(0.9, pred_x))
        pred_y = max(0.1, min(0.9, pred_y))

        return pred_x, pred_y

    def is_prediction_active(self):
        """Check if prediction is currently active"""
        return self.prediction_active


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    detection_signal = pyqtSignal(float, float, float, bool)  # x_ratio, y_ratio, confidence, is_prediction
    error_signal = pyqtSignal(str)  # For reporting errors

    def __init__(self, camera_id=1):
        super().__init__()
        self.camera_id = camera_id
        self.running = True
        self.tracking_enabled = True

        # Initialize YOLOv8
        try:
            self.model = YOLO("yolov8n.pt")  # Use the smallest model for speed
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8: {e}")
            self.model = None

        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480

        # Improved smoothing settings
        self.smoothing_factor = 0.85  # Higher smoothing for more stability
        self.prev_target_x = 0.5
        self.prev_target_y = 0.5

        # Detection confidence threshold
        self.confidence_threshold = 0.5

        # Motion predictor for prediction when target leaves frame
        self.motion_predictor = MotionPredictor()

        # Flag to indicate we want to target the chest, not center
        self.target_chest = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Get actual frame dimensions
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_without_detection = 0
        prediction_active = False

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            display_frame = frame.copy()

            if self.tracking_enabled and self.model is not None:
                # Detect people with YOLOv8
                results = self.model(frame, classes=0)  # Class 0 is person

                # Find the best detection
                best_box = None
                best_confidence = 0

                if len(results) > 0:
                    # Process detection results
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            confidence = float(box.conf[0])

                            # Only process high confidence detections
                            if confidence > self.confidence_threshold:
                                # Draw rectangle
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_frame, f"{confidence:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                # Track person with highest confidence
                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_box = (x1, y1, x2, y2)

                # If we found a person, track them
                if best_box is not None:
                    frames_without_detection = 0
                    prediction_active = False

                    x1, y1, x2, y2 = best_box

                    # Calculate target position - use chest area if enabled
                    if self.target_chest:
                        # Target the upper-middle portion of the body (chest area)
                        # Chest is approximately at 30% of the way down from the top of the bounding box
                        chest_y = y1 + int(0.3 * (y2 - y1))
                        center_x = (x1 + x2) // 2

                        # Mark chest target
                        cv2.circle(display_frame, (center_x, chest_y), 8, (0, 0, 255), -1)

                        # Calculate normalized coordinates
                        target_x = center_x / self.frame_width
                        target_y = chest_y / self.frame_height
                    else:
                        # Use center of bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Calculate normalized coordinates
                        target_x = center_x / self.frame_width
                        target_y = center_y / self.frame_height

                    # Apply stronger smoothing for more stability
                    target_x = self.smoothing_factor * self.prev_target_x + (1 - self.smoothing_factor) * target_x
                    target_y = self.smoothing_factor * self.prev_target_y + (1 - self.smoothing_factor) * target_y

                    self.prev_target_x = target_x
                    self.prev_target_y = target_y

                    # Update motion predictor with new position
                    self.motion_predictor.add_position(target_x, target_y)

                    # Emit detection signal
                    self.detection_signal.emit(target_x, target_y, best_confidence, False)

                    # Draw crosshair
                    target_pixel_x = int(self.prev_target_x * self.frame_width)
                    target_pixel_y = int(self.prev_target_y * self.frame_height)
                    cv2.drawMarker(display_frame, (target_pixel_x, target_pixel_y), (0, 0, 255),
                                   cv2.MARKER_CROSS, 20, 2)

                    # Draw targeting info
                    cv2.putText(display_frame, "TARGET LOCKED", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # No detection - use prediction if enabled
                    frames_without_detection += 1

                    # Start prediction after a few missed frames
                    if frames_without_detection > 5 and not prediction_active:
                        prediction_active = self.motion_predictor.start_prediction()

                    if prediction_active and self.motion_predictor.is_prediction_active():
                        # Get predicted position
                        pred_x, pred_y = self.motion_predictor.get_predicted_position()

                        # Emit predicted position
                        self.detection_signal.emit(pred_x, pred_y, 0.5, True)  # Using 0.5 as confidence for predictions

                        # Draw predicted position indicator
                        pred_pixel_x = int(pred_x * self.frame_width)
                        pred_pixel_y = int(pred_y * self.frame_height)
                        cv2.circle(display_frame, (pred_pixel_x, pred_pixel_y), 15, (0, 165, 255), 2)
                        cv2.drawMarker(display_frame, (pred_pixel_x, pred_pixel_y), (0, 165, 255),
                                       cv2.MARKER_CROSS, 15, 2)

                        # Draw prediction info
                        cv2.putText(display_frame, "PREDICTION ACTIVE", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    elif prediction_active and not self.motion_predictor.is_prediction_active():
                        # Prediction timed out
                        prediction_active = False

            # Emit the processed frame
            self.change_pixmap_signal.emit(display_frame)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class ContinuousMovementController:
    """Controller for smooth, continuous pan/tilt movement"""

    def __init__(self, deadzone=0.08, max_speed=63):
        self.deadzone = deadzone  # Deadzone around center where no movement occurs
        self.max_speed = max_speed
        self.min_speed = 8  # Minimum speed to ensure movement

        # Current movement state
        self.current_pan_direction = 0  # -1: left, 0: none, 1: right
        self.current_tilt_direction = 0  # -1: up, 0: none, 1: down
        self.current_pan_speed = 0
        self.current_tilt_speed = 0

    def calculate_movement(self, x_error, y_error):
        """Calculate continuous movement based on target error"""
        # Determine pan direction and speed
        if abs(x_error) < self.deadzone:
            pan_direction = 0
            pan_speed = 0
        else:
            pan_direction = 1 if x_error > 0 else -1
            error_magnitude = abs(x_error)

            # Calculate speed based on distance from center
            if error_magnitude > 0.3:
                pan_speed = self.max_speed
            elif error_magnitude > 0.2:
                pan_speed = int(self.max_speed * 0.7)
            else:
                pan_speed = self.min_speed

        # Determine tilt direction and speed
        if abs(y_error) < self.deadzone:
            tilt_direction = 0
            tilt_speed = 0
        else:
            tilt_direction = 1 if y_error > 0 else -1
            error_magnitude = abs(y_error)

            # Calculate speed based on distance from center
            if error_magnitude > 0.3:
                tilt_speed = self.max_speed
            elif error_magnitude > 0.2:
                tilt_speed = int(self.max_speed * 0.7)
            else:
                tilt_speed = self.min_speed

        # Update current movement state
        self.current_pan_direction = pan_direction
        self.current_tilt_direction = tilt_direction
        self.current_pan_speed = pan_speed
        self.current_tilt_speed = tilt_speed

        # Determine if we need to prioritize pan or tilt
        if pan_speed > 0 and tilt_speed > 0:
            # Both movements needed, prioritize based on error magnitude
            if abs(x_error) > abs(y_error):
                # Prioritize pan
                return pan_speed, pan_direction, 0, 0
            else:
                # Prioritize tilt
                return 0, 0, tilt_speed, tilt_direction
        else:
            # Return both movements
            return pan_speed, pan_direction, tilt_speed, tilt_direction


class ArduinoLaserController:
    def __init__(self, port='COM4', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connected = False

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            self.connected = True
            print(f"Connected to Arduino at {port}")
            time.sleep(2)  # Allow Arduino to reset
        except Exception as e:
            print(f"Failed to connect to Arduino: {str(e)}")

    def toggle_laser(self, state):
        """Turn laser on or off"""
        if not self.connected:
            return False

        try:
            if state:
                self.ser.write(b'LON\n')
            else:
                self.ser.write(b'LOFF\n')
            return True
        except Exception as e:
            print(f"Error controlling laser: {str(e)}")
            return False

    def close(self):
        """Close the serial connection"""
        if self.ser and self.ser.is_open:
            self.toggle_laser(False)  # Turn off laser before closing
            self.ser.close()


class HumanTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Human Tracking System")
        self.setGeometry(100, 100, 1000, 700)

        # Pan/Tilt parameters
        self.ADDRESS = 49
        self.MAX_SPEED = 0x3F  # Maximum speed (63)
        self.current_speed = self.MAX_SPEED // 2
        self.ser = None

        # Tracking parameters
        self.tracking_enabled = True
        self.laser_enabled = False
        self.auto_tracking = False
        self.auto_laser = True  # Auto-enable laser when tracking
        self.target_x = 0.5
        self.target_y = 0.5
        self.prediction_enabled = True  # Enable trajectory prediction when target leaves frame
        self.is_predicted_position = False  # Flag to indicate if current position is predicted

        # Movement parameters
        self.last_movement_update = time.time()
        self.movement_update_interval = 0.1  # Seconds between movement updates

        # Last sent movement command
        self.last_command = None

        # Setup continuous movement controller
        self.movement_controller = ContinuousMovementController(deadzone=0.08)

        # Active tracking state
        self.is_tracking_active = False
        self.tracking_timeout = 2.0  # Seconds before tracking is considered inactive
        self.last_detection_time = 0

        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Camera feed panel
        self.setup_camera_panel()

        # Controls panel
        self.controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_panel)
        self.main_layout.addWidget(self.controls_panel, 1)

        # Status display
        self.status_label = QLabel("Status: Initializing...")
        self.controls_layout.addWidget(self.status_label)

        # Connect to devices
        self.connect_device()
        self.laser_controller = ArduinoLaserController()

        # Initialize video thread
        self.video_thread = VideoThread()

        # Setup controls
        self.setup_control_panel()
        self.setup_tracking_panel()
        self.setup_laser_panel()

        # Connect video thread signals
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.detection_signal.connect(self.handle_detection)
        self.video_thread.error_signal.connect(self.handle_error)
        self.video_thread.start()

        # Timer for continuous tracking and movement updates
        self.tracking_timer = QTimer()
        self.tracking_timer.timeout.connect(self.update_tracking)
        self.tracking_timer.start(50)  # Update every 50ms for responsive tracking

    def setup_camera_panel(self):
        """Setup the camera feed panel"""
        self.camera_panel = QWidget()
        self.camera_layout = QVBoxLayout(self.camera_panel)
        self.main_layout.addWidget(self.camera_panel, 2)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Waiting for camera...")
        self.camera_layout.addWidget(self.image_label)

    def setup_control_panel(self):
        """Setup the pan/tilt control panel"""
        control_group = QGroupBox("Pan/Tilt Control")
        control_layout = QGridLayout()

        # Movement buttons
        self.pan_right_btn = QPushButton("Pan Right")
        self.pan_right_btn.pressed.connect(lambda: self.start_movement(self.pan_right))
        self.pan_right_btn.released.connect(self.stop_movement)
        control_layout.addWidget(self.pan_right_btn, 1, 2)

        self.pan_left_btn = QPushButton("Pan Left")
        self.pan_left_btn.pressed.connect(lambda: self.start_movement(self.pan_left))
        self.pan_left_btn.released.connect(self.stop_movement)
        control_layout.addWidget(self.pan_left_btn, 1, 0)

        self.tilt_up_btn = QPushButton("Tilt Up")
        self.tilt_up_btn.pressed.connect(lambda: self.start_movement(self.tilt_up))
        self.tilt_up_btn.released.connect(self.stop_movement)
        control_layout.addWidget(self.tilt_up_btn, 0, 1)

        self.tilt_down_btn = QPushButton("Tilt Down")
        self.tilt_down_btn.pressed.connect(lambda: self.start_movement(self.tilt_down))
        self.tilt_down_btn.released.connect(self.stop_movement)
        control_layout.addWidget(self.tilt_down_btn, 2, 1)

        # Stop button
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.stop_btn.clicked.connect(self.stop_movement)
        control_layout.addWidget(self.stop_btn, 1, 1)

        # Speed control
        speed_layout = QVBoxLayout()
        speed_label = QLabel("Movement Speed:")
        speed_layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(5)
        self.speed_slider.setMaximum(self.MAX_SPEED)
        self.speed_slider.setValue(self.current_speed)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)

        self.speed_value_label = QLabel(f"Speed: {self.current_speed}")
        speed_layout.addWidget(self.speed_value_label)

        control_layout.addLayout(speed_layout, 3, 0, 1, 3)

        # Home position button
        self.home_btn = QPushButton("Go Home")
        self.home_btn.setStyleSheet("background-color: blue; color: white;")
        self.home_btn.clicked.connect(self.go_to_home_position)
        control_layout.addWidget(self.home_btn, 4, 1)

        control_group.setLayout(control_layout)
        self.controls_layout.addWidget(control_group)

    def setup_tracking_panel(self):
        """Setup the tracking control panel"""
        tracking_group = QGroupBox("Human Tracking Controls")
        tracking_layout = QVBoxLayout()

        # Enable/disable tracking
        self.tracking_checkbox = QCheckBox("Enable Human Detection")
        self.tracking_checkbox.setChecked(True)
        self.tracking_checkbox.stateChanged.connect(self.toggle_tracking)
        tracking_layout.addWidget(self.tracking_checkbox)

        # Auto-tracking
        self.auto_tracking_checkbox = QCheckBox("Enable Auto-Tracking")
        self.auto_tracking_checkbox.setChecked(False)
        self.auto_tracking_checkbox.stateChanged.connect(self.toggle_auto_tracking)
        tracking_layout.addWidget(self.auto_tracking_checkbox)

        # Enable prediction
        self.prediction_checkbox = QCheckBox("Continue Tracking When Target Exits Frame")
        self.prediction_checkbox.setChecked(True)
        self.prediction_checkbox.stateChanged.connect(self.toggle_prediction)
        tracking_layout.addWidget(self.prediction_checkbox)

        # Target chest
        self.chest_targeting_checkbox = QCheckBox("Target Chest Instead of Center")
        self.chest_targeting_checkbox.setChecked(True)
        self.chest_targeting_checkbox.stateChanged.connect(self.toggle_chest_targeting)
        tracking_layout.addWidget(self.chest_targeting_checkbox)

        # Continuous movement mode
        self.continuous_movement_checkbox = QCheckBox("Use Continuous Movement")
        self.continuous_movement_checkbox.setChecked(True)
        tracking_layout.addWidget(self.continuous_movement_checkbox)

        # Tracking sensitivity (deadzone)
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Deadzone:"))

        self.deadzone_slider = QSlider(Qt.Horizontal)
        self.deadzone_slider.setMinimum(5)
        self.deadzone_slider.setMaximum(20)
        self.deadzone_slider.setValue(int(self.movement_controller.deadzone * 100))
        self.deadzone_slider.valueChanged.connect(self.update_deadzone)
        sens_layout.addWidget(self.deadzone_slider)

        self.deadzone_label = QLabel(f"{int(self.movement_controller.deadzone * 100)}%")
        sens_layout.addWidget(self.deadzone_label)
        tracking_layout.addLayout(sens_layout)

        # Smoothing control
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Movement Smoothing:"))

        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setMinimum(50)
        self.smoothing_slider.setMaximum(95)
        self.smoothing_slider.setValue(int(self.video_thread.smoothing_factor * 100))
        self.smoothing_slider.valueChanged.connect(self.update_smoothing)
        smooth_layout.addWidget(self.smoothing_slider)

        self.smoothing_label = QLabel(f"{int(self.video_thread.smoothing_factor * 100)}%")
        smooth_layout.addWidget(self.smoothing_label)
        tracking_layout.addLayout(smooth_layout)

        # Movement update interval
        update_layout = QHBoxLayout()
        update_layout.addWidget(QLabel("Movement Update Rate:"))

        self.update_slider = QSlider(Qt.Horizontal)
        self.update_slider.setMinimum(5)
        self.update_slider.setMaximum(50)
        self.update_slider.setValue(int(self.movement_update_interval * 100))
        self.update_slider.valueChanged.connect(self.update_interval)
        update_layout.addWidget(self.update_slider)

        self.update_label = QLabel(f"{int(self.movement_update_interval * 100)}ms")
        update_layout.addWidget(self.update_label)
        tracking_layout.addLayout(update_layout)

        tracking_group.setLayout(tracking_layout)
        self.controls_layout.addWidget(tracking_group)

    def setup_laser_panel(self):
        """Setup the laser control panel"""
        laser_group = QGroupBox("Laser Control")
        laser_layout = QVBoxLayout()

        # Laser toggle button
        self.laser_btn = QPushButton("Toggle Laser")
        self.laser_btn.setCheckable(True)
        self.laser_btn.setChecked(False)
        self.laser_btn.clicked.connect(self.toggle_laser)
        laser_layout.addWidget(self.laser_btn)

        # Auto laser when tracking
        self.auto_laser_checkbox = QCheckBox("Auto-enable laser when tracking")
        self.auto_laser_checkbox.setChecked(True)
        self.auto_laser_checkbox.stateChanged.connect(self.toggle_auto_laser)
        laser_layout.addWidget(self.auto_laser_checkbox)

        laser_group.setLayout(laser_layout)
        self.controls_layout.addWidget(laser_group)

    def handle_error(self, error_message):
        """Handle error messages"""
        self.status_label.setText(f"Error: {error_message}")

    def connect_device(self):
        """Connect to the pan/tilt controller"""
        try:
            self.ser = serial.Serial(
                port='COM3',  # Change to your actual COM port
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.status_label.setText(f"Connected to COM3 | Address: {self.ADDRESS} | Speed: {self.current_speed}")
        except Exception as e:
            self.status_label.setText(f"Pan/Tilt connection error: {str(e)}")

    def update_image(self, cv_img):
        """Update the image_label with a new OpenCV image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from OpenCV image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def handle_detection(self, x_ratio, y_ratio, confidence, is_prediction):
        """Handle a new detection or prediction"""
        self.target_x = x_ratio
        self.target_y = y_ratio
        self.is_predicted_position = is_prediction

        # Update tracking activity state
        self.is_tracking_active = True
        self.last_detection_time = time.time()

        # Automatically enable laser if auto-laser is on (only for real detections)
        if self.auto_laser and self.auto_tracking and not self.laser_enabled and not is_prediction:
            self.toggle_laser(True)
            self.laser_btn.setChecked(True)

    def update_tracking(self):
        """Update the tracking and movement"""
        # Check if tracking is still active
        if time.time() - self.last_detection_time > self.tracking_timeout:
            if self.is_tracking_active:
                self.is_tracking_active = False
                # Turn off laser when tracking is lost
                if self.auto_laser and self.laser_enabled:
                    self.toggle_laser(False)
                    self.laser_btn.setChecked(False)
                # Stop movement when tracking is completely lost
                self.stop_movement()

        if not self.auto_tracking or not self.is_tracking_active:
            return

        # Check if it's time to update movement
        current_time = time.time()
        if current_time - self.last_movement_update < self.movement_update_interval:
            return

        self.last_movement_update = current_time

        # Calculate how far the target is from the center
        x_error = self.target_x - 0.5
        y_error = self.target_y - 0.5

        if self.continuous_movement_checkbox.isChecked():
            # Use continuous movement
            self.perform_continuous_movement(x_error, y_error)
        else:
            # Use step-based movement
            self.perform_step_movement(x_error, y_error)

    def perform_continuous_movement(self, x_error, y_error):
        """Perform continuous movement based on target position"""
        # Calculate movement using the controller
        pan_speed, pan_direction, tilt_speed, tilt_direction = self.movement_controller.calculate_movement(x_error,
                                                                                                           y_error)

        # Execute the movement command
        if pan_speed > 0:
            # Pan movement needed
            self.current_speed = pan_speed
            if pan_direction > 0:
                # Pan right
                self.execute_continuous_movement(self.pan_right)
            else:
                # Pan left
                self.execute_continuous_movement(self.pan_left)
        elif tilt_speed > 0:
            # Tilt movement needed
            self.current_speed = tilt_speed
            if tilt_direction > 0:
                # Tilt down
                self.execute_continuous_movement(self.tilt_down)
            else:
                # Tilt up
                self.execute_continuous_movement(self.tilt_up)
        else:
            # No movement needed, stop
            self.stop_movement()

    def execute_continuous_movement(self, movement_func):
        """Execute continuous movement command, only if different from last command"""
        try:
            if self.ser and self.ser.is_open:
                # Generate new command
                command = movement_func()

                # Only send command if it's different from the last one
                if command != self.last_command:
                    self.ser.write(command)
                    self.last_command = command
                    self.status_label.setText(f"Moving: {[hex(b) for b in command]}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def perform_step_movement(self, x_error, y_error):
        """Perform step-based movement (legacy method)"""
        # Calculate pan movement
        if abs(x_error) < self.movement_controller.deadzone:
            pan_speed = 0
        else:
            pan_direction = 1 if x_error > 0 else -1
            error_magnitude = abs(x_error)

            if error_magnitude > 0.3:
                pan_speed = self.MAX_SPEED
            elif error_magnitude > 0.2:
                pan_speed = int(self.MAX_SPEED * 0.7)
            else:
                pan_speed = self.movement_controller.min_speed

            # Set direction
            if pan_direction < 0:
                pan_speed = -pan_speed

        # Calculate tilt movement
        if abs(y_error) < self.movement_controller.deadzone:
            tilt_speed = 0
        else:
            tilt_direction = 1 if y_error > 0 else -1
            error_magnitude = abs(y_error)

            if error_magnitude > 0.3:
                tilt_speed = self.MAX_SPEED
            elif error_magnitude > 0.2:
                tilt_speed = int(self.MAX_SPEED * 0.7)
            else:
                tilt_speed = self.movement_controller.min_speed

            # Set direction
            if tilt_direction < 0:
                tilt_speed = -tilt_speed

        # Execute movement if needed
        if abs(pan_speed) > 0:
            self.current_speed = abs(pan_speed)
            if pan_speed > 0:
                self.execute_step(self.pan_right)
            else:
                self.execute_step(self.pan_left)
        elif abs(tilt_speed) > 0:
            self.current_speed = abs(tilt_speed)
            if tilt_speed > 0:
                self.execute_step(self.tilt_down)
            else:
                self.execute_step(self.tilt_up)
        else:
            # Stop if no movement needed
            self.stop_movement()

    def execute_step(self, movement_func):
        """Execute a single step movement"""
        try:
            if self.ser and self.ser.is_open:
                # Send movement command
                command = movement_func()
                self.ser.write(command)

                # Wait for a short duration
                time.sleep(0.05)

                # Then stop movement
                self.stop_movement()
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def update_speed(self, value):
        """Update the movement speed"""
        self.current_speed = value
        self.speed_value_label.setText(f"Speed: {self.current_speed}")
        self.movement_controller.max_speed = value

    def update_deadzone(self, value):
        """Update the tracking deadzone"""
        self.movement_controller.deadzone = value / 100.0
        self.deadzone_label.setText(f"{value}%")

    def update_smoothing(self, value):
        """Update tracking smoothing factor"""
        smoothing = value / 100.0
        self.smoothing_label.setText(f"{value}%")
        self.video_thread.smoothing_factor = smoothing

    def update_interval(self, value):
        """Update movement update interval"""
        self.movement_update_interval = value / 100.0
        self.update_label.setText(f"{value}ms")

    def toggle_tracking(self, state):
        """Toggle human detection"""
        self.tracking_enabled = state == Qt.Checked
        self.video_thread.tracking_enabled = self.tracking_enabled

        # Turn off laser if tracking is disabled
        if not self.tracking_enabled and self.auto_laser and self.laser_enabled:
            self.toggle_laser(False)
            self.laser_btn.setChecked(False)

    def toggle_auto_tracking(self, state):
        """Toggle auto-tracking mode"""
        self.auto_tracking = state == Qt.Checked

        # Turn off laser if auto-tracking is disabled
        if not self.auto_tracking and self.auto_laser and self.laser_enabled:
            self.toggle_laser(False)
            self.laser_btn.setChecked(False)

    def toggle_prediction(self, state):
        """Toggle prediction when target exits frame"""
        self.prediction_enabled = state == Qt.Checked

    def toggle_chest_targeting(self, state):
        """Toggle targeting chest vs center of bounding box"""
        self.video_thread.target_chest = state == Qt.Checked

    def toggle_auto_laser(self, state):
        """Toggle auto laser mode"""
        self.auto_laser = state == Qt.Checked

        # Turn off laser if auto-laser is disabled
        if not self.auto_laser and self.laser_enabled:
            self.toggle_laser(False)
            self.laser_btn.setChecked(False)

    def toggle_laser(self, state):
        """Toggle the laser on/off"""
        self.laser_enabled = state
        if self.laser_controller:
            success = self.laser_controller.toggle_laser(state)
            if not success:
                self.status_label.setText("Error controlling laser")
                self.laser_btn.setChecked(not state)  # Revert the button state
                self.laser_enabled = not state
            else:
                self.status_label.setText(f"Laser {'ON' if state else 'OFF'}")

    def pelco_d_command(self, command1, command2, data1, data2):
        """Generate Pelco-D protocol command."""
        msg = bytearray([0xFF, self.ADDRESS, command1, command2, data1, data2])
        checksum = sum(msg[1:]) % 256
        msg.append(checksum)
        return msg

    def pan_right(self):
        """Pan right command."""
        return self.pelco_d_command(0x00, 0x02, self.current_speed, 0x00)

    def pan_left(self):
        """Pan left command."""
        return self.pelco_d_command(0x00, 0x04, self.current_speed, 0x00)

    def tilt_up(self):
        """Tilt up command."""
        return self.pelco_d_command(0x00, 0x08, 0x00, self.current_speed)

    def tilt_down(self):
        """Tilt down command."""
        return self.pelco_d_command(0x00, 0x10, 0x00, self.current_speed)

    def stop(self):
        """Stop all movement."""
        return self.pelco_d_command(0x00, 0x00, 0x00, 0x00)

    def start_movement(self, movement_func):
        """Start a movement."""
        try:
            if self.ser and self.ser.is_open:
                command = movement_func()
                self.ser.write(command)
                self.last_command = command
                self.status_label.setText(f"Command sent: {[hex(b) for b in command]}")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def stop_movement(self):
        """Stop all movement."""
        try:
            if self.ser and self.ser.is_open:
                command = self.stop()
                self.ser.write(command)
                self.last_command = None
                self.status_label.setText("Stopped")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def go_to_home_position(self):
        """Move pan/tilt to home position (center/0 position)"""
        try:
            if self.ser and self.ser.is_open:
                self.status_label.setText("Moving to home position...")

                # First stop any current movement
                stop_cmd = self.stop()
                self.ser.write(stop_cmd)
                time.sleep(0.5)

                # Use medium speed
                original_speed = self.current_speed
                self.current_speed = 30

                # First move all the way left/up to establish a reference
                # (this assumes the pan/tilt has physical limits)
                max_movement_time = 8  # Maximum time in seconds for movement

                # Move left
                left_cmd = self.pan_left()
                self.ser.write(left_cmd)
                time.sleep(max_movement_time)
                self.ser.write(stop_cmd)
                time.sleep(0.5)

                # Move up
                up_cmd = self.tilt_up()
                self.ser.write(up_cmd)
                time.sleep(max_movement_time)
                self.ser.write(stop_cmd)
                time.sleep(0.5)

                # Now move right and down for half the time to reach approximate center
                right_cmd = self.pan_right()
                self.ser.write(right_cmd)
                time.sleep(max_movement_time / 2)
                self.ser.write(stop_cmd)
                time.sleep(0.5)

                down_cmd = self.tilt_down()
                self.ser.write(down_cmd)
                time.sleep(max_movement_time / 2)
                self.ser.write(stop_cmd)

                # Restore original speed
                self.current_speed = original_speed

                self.status_label.setText("Moved to home position")

        except Exception as e:
            self.status_label.setText(f"Error moving to home: {str(e)}")

    def closeEvent(self, event):
        """Clean up when closing."""
        # First turn off laser
        if self.laser_controller:
            self.laser_controller.toggle_laser(False)

        # Move to home position
        self.go_to_home_position()

        # Stop the video thread
        self.video_thread.stop()

        # Close all connections
        if self.laser_controller:
            self.laser_controller.close()

        # Close the serial connection to pan/tilt
        if self.ser and self.ser.is_open:
            self.ser.close()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HumanTrackerApp()
    window.show()
    sys.exit(app.exec_())
