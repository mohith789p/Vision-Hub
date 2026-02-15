# Visual Interpreter

Real-time image captioning system using pre-trained vision-language models.

## Overview

This project generates automatic captions for live video feeds using **Microsoft GIT (Generative Image-to-Text)** model. It processes webcam frames and provides natural language descriptions of visual content in real-time.

## Features

- **Real-time Captioning**: Generates captions for live video feed
- **Pre-trained Model**: Uses Microsoft GIT-base model trained on COCO dataset
- **Frame Skipping**: Optimized for performance with configurable frame intervals
- **Text Wrapping**: Displays captions with proper text wrapping on screen
- **Easy Customization**: Adjustable caption length and frame processing rate
- **GPU Support**: Compatible with CUDA for faster inference

## Requirements

- Python 3.8+
- OpenCV
- Hugging Face Transformers
- PyTorch
- CUDA (optional, for GPU acceleration)

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

## How It Works

1. Loads Microsoft GIT-base model and processor from Hugging Face
2. Opens default webcam
3. Captures video frames
4. Processes every nth frame (frame_skip) for performance
5. Generates caption using transformer model
6. Wraps text to fit display width
7. Renders caption on the video frame
8. Displays output in real-time window
9. Exits on 'q' key press

## Configuration

Modify these parameters in `main.py`:

```python
# Change frame skip rate (higher = faster but less frequent updates)
frame_skip = 8  # Process every 8th frame

# Adjust caption length
max_new_tokens = 20  # Number of tokens in caption

# Text wrapping width
textwrap.wrap(caption, 40)  # Characters per line
```

## Model Information

**Microsoft GIT (Generative Image-to-Text)**

- **Model Size**: ~350 MB
- **Training Data**: COCO dataset with 118K images
- **Architecture**: Vision encoder + Language decoder
- **Input**: Images (224x224 recommended)
- **Output**: Natural language captions

Model card: https://huggingface.co/microsoft/git-base-coco

## Performance Optimization

### Frame Skip Strategy

- `frame_skip = 4`: More frequent captions, higher CPU usage
- `frame_skip = 8`: Balanced (default)
- `frame_skip = 16`: Faster processing, less frequent updates

### Reduce Inference Time

```python
# Reduce token generation
max_new_tokens = 10  # Shorter captions

# Adjust image resolution
# Smaller images = faster inference
```

### GPU Acceleration

Enable GPU if available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

## Performance Tips

- Ensure good lighting for accurate descriptions
- Keep objects clearly visible
- Adjust frame_skip based on desired update frequency
- Use GPU for real-time performance
- Close other applications to free up system resources

## Use Cases

- **Accessibility**: Audio descriptions for visually impaired users
- **Content Analysis**: Automatic video tagging and annotation
- **Research**: Understanding model capabilities
- **Augmented Reality**: Real-time scene understanding
- **Surveillance**: Scene description and anomaly detection

## Troubleshooting

### Out of Memory Error

- Increase frame_skip for less frequent processing
- Reduce max_new_tokens
- Use GPU with sufficient VRAM
- Close other applications

### Slow Performance

- Increase frame_skip value
- Reduce max_new_tokens
- Enable GPU acceleration
- Lower video resolution if possible

### Poor Captions

- Ensure adequate lighting
- Center objects in frame
- Increase max_new_tokens for more detailed descriptions
- Train custom model for specific domain

## Future Enhancements

- Multi-language support
- Custom model fine-tuning
- Video summarization
- Real-time translation of captions
- Integration with text-to-speech
- Support for multiple image regions (Dense Captioning)
- Integration with other vision models
