# Object Detection

Real-time object detection using YOLOv8 deep learning model.

## Overview

This project performs real-time object detection using **YOLOv8** (You Only Look Once v8), a state-of-the-art deep learning model. It processes video feed from a webcam and detects hundreds of object classes with accurate bounding boxes and confidence scores.

## Features

- **Real-time Detection**: YOLOv8 inference on live webcam feed
- **High Accuracy**: Pre-trained model on COCO dataset with 80+ object classes
- **Multiple Model Variants**: Supports nano (n), small (s), medium (m), large (l), xlarge (xl)
- **Confidence Filtering**: Configurable confidence threshold (default: 0.5)
- **Visual Output**: Displays bounding boxes, labels, and confidence scores
- **Easy Integration**: Simple and intuitive API

## Requirements

- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- PyTorch (automatically installed with Ultralytics)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

Press `q` to quit the application.

## Included Models

- `yolo26n.pt`: Lightweight nano model (fastest, less accurate)
- `yolov8n.pt`: Official YOLOv8 nano model

## How It Works

1. Loads pre-trained YOLOv8 model
2. Opens default webcam (index 0)
3. Captures video frames
4. Runs inference on each frame
5. Filters detections by confidence threshold
6. Draws bounding boxes and labels on frame
7. Displays result in real-time
8. Exits on 'q' key press

## Configuration

Modify these parameters in `main.py`:

```python
model = YOLO("object-detection/yolov8n.pt")  # Change model variant
results = model(frame, conf=0.5, stream=True)  # Adjust confidence threshold
```

### Model Variants Comparison

| Model       | Size     | Speed   | Accuracy | Use Case          |
| ----------- | -------- | ------- | -------- | ----------------- |
| nano (n)    | ~3.2 MB  | Fastest | Lower    | Edge devices      |
| small (s)   | ~11.6 MB | Fast    | Medium   | Real-time apps    |
| medium (m)  | ~49.7 MB | Medium  | High     | Balanced          |
| large (l)   | ~94.9 MB | Slower  | Higher   | Accuracy-critical |
| xlarge (xl) | ~176 MB  | Slowest | Highest  | Maximum precision |

## Detectable Objects

YOLOv8 can detect 80 classes including:

- People, animals (dog, cat, bird, etc.)
- Vehicles (car, truck, motorcycle, bus, etc.)
- Electronics (laptop, phone, TV, etc.)
- Food (apple, banana, orange, etc.)
- Sports items, furniture, and more

See detailed class list: https://github.com/ultralytics/yolov8

## Performance Tips

- Use nano or small models for real-time performance on CPU
- Use GPU for faster inference with larger models
- Adjust confidence threshold based on use case
- Lower confidence for detecting distant objects
- Higher confidence to reduce false positives

## Troubleshooting

### Slow Performance

- Use a smaller model variant (nano or small)
- Reduce frame resolution
- Use GPU acceleration if available

### Poor Detection

- Improve lighting
- Adjust confidence threshold lower
- Use a larger model variant

### Webcam Issues

- Verify webcam is connected
- Check camera permissions
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

## Future Enhancements

- Custom model training
- Track objects across frames
- Export detections to JSON/CSV
- Support for video file input
- GPU acceleration with CUDA
- Integration with external pipelines
