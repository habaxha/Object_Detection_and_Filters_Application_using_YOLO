# YOLO Object Detection and Filters Application

This application is a powerful tool for image and video processing, combining YOLO (You Only Look Once) object detection with various image filters. It provides a user-friendly interface for processing images, videos, and live video streams.

## Features

- YOLO Object Detection
- Image Processing
- Video Processing
- Live Video Processing
- Multiple Image Filters:
  - Edge Detection
  - Sharpen
  - Gaussian Blur
  - Brightness Adjustment
  - Erosion
  - Dilation
  - Sepia Tone
  - Contrast Adjustment
  - Negative
  - Emboss

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- tkinter

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/yolo-object-detection-app.git
   ```

2. Install the required packages:
   ```
   pip install opencv-python numpy pillow
   ```

3. Download the YOLO weights, cfg, and names files and place them in the project directory.

## Usage

1. Run the application:
   ```
   python Object_Detection_and_Filters_Application_using_YOLO.py
   ```

2. Load the YOLO model:
   - Click "Select Weights" to choose the .weights file
   - Click "Select CFG" to choose the .cfg file
   - Click "Select Names" to choose the .names file

3. Choose a processing mode:
   - Image Processing
   - Video Processing
   - Live Video Processing

4. Use the buttons on the right to apply filters or toggle object detection

5. Use the "Minimize/Maximize" and "Close" buttons to manage the display

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- YOLO: Real-Time Object Detection
- OpenCV library
- NumPy library
- Pillow library
