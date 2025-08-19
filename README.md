Human Tracking System with Laser Targeting
A computer vision-based system that automatically detects, tracks, and targets humans using a pan/tilt mount and laser pointer. The system uses YOLOv8 for real-time human detection and can track subjects even when they temporarily leave the camera's field of view.

Overview
This project integrates several technologies to create an automated human tracking system:

Real-time human detection using YOLOv8
Pan/tilt control system for camera/laser positioning
Arduino-controlled laser targeting
Qt-based user interface with video monitoring
Resource monitoring to ensure system stability
![System Demo](docs/images/system_demo.jpg)

Hardware Components
Pan/Tilt Unit: Outdoor Pantilt model 3050DZ
Camera: USB webcam with minimum 640x480 resolution
Arduino: Arduino Uno or compatible board for laser control
Laser Module: 5V laser diode module (3-pin type)
Computer: System with GPU recommended for optimal performance
Hardware Setup
Mount the camera on the pan/tilt unit
Connect the laser module to the Arduino (Signal pin to D9)
Connect the pan/tilt unit to the control computer via serial (RS-485 to USB)
Connect the Arduino to the computer via USB
Connect the camera to the computer via USB
Software Requirements
Python 3.8 or higher
OpenCV
PyQt5
Ultralytics YOLOv8
PySerial
NumPy
Installation
Clone this repository:
bash
git clone https://github.com/OmarAlshaibani/human-tracking-system-with-Outdoor-Pan-Tilt.git
cd human-tracking-system-with-Outdoor-Pan-Tilt
cd human-tracking-system-with-Outdoor-Pan-Tilt
Install the required Python packages:
bash
pip install -r requirements.txt
Download the YOLOv8 model:
bash
# The system will automatically download YOLOv8n on first run
# or you can manually download it from Ultralytics
Upload the Arduino code:
Open the laser_controller.ino file in the Arduino IDE
Connect your Arduino board
Upload the sketch
Usage
Connect all hardware components
Run the main program:
bash
python human_tracker.py
The system interface allows you to:
Enable/disable human detection
Enable/disable auto-tracking
Control the pan/tilt unit manually
Toggle the laser pointer
Adjust tracking sensitivity and movement speed
Configure targeting options (chest vs. center targeting)
Common Issues and Solutions
System Crashes After Running for a Minute
Problem: The system would crash after running for about a minute, especially on systems with limited resources.

Solution: Implemented resource monitoring that:

Tracks CPU and memory usage in real-time
Automatically adjusts frame processing rate when resources are constrained
Performs garbage collection when memory usage exceeds thresholds
Provides visual feedback on resource usage through the UI

Pan/Tilt Control Sensitivity
Problem: The pan/tilt movement was either too aggressive or too slow to effectively track subjects.

Solution:

Added adjustable deadzone setting to prevent micro-movements
Implemented variable speed control based on target distance
Added smoothing controls to reduce jittery movements
Created continuous movement mode for smoother tracking
Configuration
The system is highly configurable through the UI. Key settings include:

Detection Confidence: Minimum confidence threshold for human detection
Tracking Deadzone: How far a target must move before tracking adjusts
Movement Smoothing: How smoothly the system follows target movements
Targeting Mode: Choose between targeting the chest or center of detected humans
File Structure
Code
human-tracking-system/
├── human_tracker.py        # Main Python application
├── laser_controller.ino    # Arduino code for laser control
├── requirements.txt        # Python dependencies
└── README.md               # This file
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Ultralytics YOLOv8 for object detection
PyQt5 for the UI framework
All contributors and testers who helped refine the system
Author
Omar Alshaibani
Last Updated: 2025-08-19
