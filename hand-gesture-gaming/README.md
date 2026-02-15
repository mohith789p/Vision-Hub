# Hand Gesture Gaming

A real-time hand gesture recognition system for gaming control using computer vision.

## Overview

This project uses **MediaPipe Hand Tracking** to detect hand gestures from a webcam and translates them into keyboard commands for gaming. Players can control game characters using hand movements and swipes.

## Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- **Gesture Recognition**: Detects hand swoop directions (left/right) to trigger keyboard inputs
- **Webcam Integration**: Processes live video feed from the default camera
- **Low Latency**: Optimized for responsive gaming control

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- PyInput (for keyboard control)
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

### Controls

- **Swipe Right**: Move Right (Arrow Key Right)
- **Swipe Left**: Move Left (Arrow Key Left)
- **Swipe Up**: Additional gesture (customizable)
- **Swipe Down**: Additional gesture (customizable)

## How It Works

1. Captures video from webcam
2. Processes each frame using MediaPipe Hand detection
3. Tracks wrist position and calculates movement deltas
4. Detects swipe direction based on movement threshold
5. Triggers corresponding keyboard input
6. Displays hand landmarks on screen

## Configuration

Adjust these parameters in `main.py` for better performance:

- `min_detection_confidence`: Hand detection confidence (default: 0.7)
- `min_tracking_confidence`: Hand tracking confidence (default: 0.7)
- Swipe threshold values (dx/dy > 50)

## Performance Tips

- Ensure good lighting for better hand detection
- Position yourself at arm's length from the camera
- Keep hands clearly visible in the frame

## Future Enhancements

- Support for both hands
- Additional gesture types (pinch, point)
- Gesture customization
- Multi-game compatibility
