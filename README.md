# Face Smile Detection

A real-time face smile detection system using OpenCV and Deep Learning. This project can detect faces in real-time video streams and classify whether the person is smiling or not.

[![Face Smile Detection Demo](https://img.youtube.com/vi/eC_GfTEylSw/0.jpg)](https://www.youtube.com/watch?v=eC_GfTEylSw)

## Features

- Real-time face detection
- Smile classification using deep learning
- FPS counter and performance monitoring
- Support for both webcam and video files
- Optimized for better performance
- Confidence score display
- Batch processing for multiple faces

## Requirements

- Python 3.6+
- OpenCV
- TensorFlow/Keras
- NumPy
- imutils

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-smile-detection.git
cd face-smile-detection
```

2. Create a virtual environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model on your own dataset:
```bash
python train.py --dataset ./datasets/smileD --model ./output/lenet.hdf5
```

### Running the Detection

For webcam detection:
```bash
python detect_smile.py --cascade haarcascade_frontalface_default.xml --model ./output/lenet.hdf5
```

For video file detection:
```bash
python detect_smile.py --cascade haarcascade_frontalface_default.xml --model ./output/lenet.hdf5 --video path_to_video.mp4
```

Additional options:
- `--skip`: Number of frames to skip (default: 2)
  ```bash
  python detect_smile.py --cascade haarcascade_frontalface_default.xml --model ./output/lenet.hdf5 --skip 3
  ```

## Controls

- Press 'q' to quit the application
- The FPS counter is displayed in the top-left corner
- Confidence scores are shown for each detection

## Project Structure

```
face-smile-detection/
├── datasets/              # Training dataset
├── output/               # Trained models
├── detect_smile.py       # Main detection script
├── train.py             # Training script
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Performance Optimization

The system includes several optimizations:
- Frame skipping for better performance
- Batch processing for multiple faces
- Optimized face detection parameters
- Efficient memory usage
- Configurable processing settings

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for face detection
- TensorFlow/Keras for deep learning
- The dataset providers
