# Media Control

A gesture-based media control system using hand recognition.

## Overview

This project enables hands-free media control using **MediaPipe Hand Tracking** to recognize finger gestures. Detect finger positions to control audio playback, volume, and other media functions through keyboard commands.

## Features

- **Real-time Hand Gesture Recognition**: Detects finger extensions and configurations
- **Gesture Classification**: Maps hand positions to media control commands
- **Angle-Based Detection**: Uses joint angles to determine finger straightness
- **Temporal Smoothing**: Implements buffer-based smoothing for stable recognition
- **Multi-Gesture Support**:
  - Play/Pause (all fingers extended)
  - Stop (closed fist)
  - Volume Up (thumb only)
  - Volume Down (thumb + index finger)

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe (v0.10.9)
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

### Gesture Commands

| Gesture              | Action      | Keyboard   |
| -------------------- | ----------- | ---------- |
| All Fingers Extended | Play/Pause  | Space      |
| Closed Fist          | Stop        | Ctrl+S     |
| Thumb Only           | Volume Up   | Up Arrow   |
| Thumb + Index        | Volume Down | Down Arrow |

## How It Works

1. Initializes MediaPipe hand detection with single hand tracking
2. Processes video frames from webcam
3. Extracts hand landmarks (21 points per hand)
4. Calculates angles between finger joints
5. Classifies gesture based on finger extension state
6. Triggers keyboard events based on gesture type
7. Uses deque buffer (length 7) for temporal smoothing

## Configuration

Adjust these parameters in `main.py`:

- `max_num_hands`: Number of hands to track (default: 1)
- `min_detection_confidence`: Detection confidence threshold (default: 0.7)
- `min_tracking_confidence`: Tracking confidence threshold (default: 0.7)
- `buffer.maxlen`: Smoothing buffer size (default: 7)
- Angle threshold: 160 degrees for finger extension

## Performance Tips

- Ensure clear lighting for accurate hand detection
- Keep hand centered in the camera view
- Move gestures slowly for better recognition
- Adjust buffer size if response feels delayed

## Customization

Modify `GestureController.classify()` method to add new gestures:

```python
def classify(self, lm):
    f = self.fingers_extended(lm)
    c = sum(f)

    if c >= 4:
        return "PlayPause"
    # Add your custom gestures here
```

## Future Enhancements

- Support for two-hand gestures
- Gesture learning/training mode
- Volume slider control
- Channel switching
- Integration with media player APIs
