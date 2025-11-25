# raspi-digit-recognition
Handwritten digit recognition on Raspberry Pi using TensorFlow Lite and Picamera2
# Handwritten Digit Recognition on Raspberry Pi  
### (CNN + TensorFlow Lite + Picamera2)

This project recognizes handwritten digits (0–9) from a sheet of paper using a **Raspberry Pi** and **camera**.  
You write a digit, hold it in front of the camera, press `c`, and the Pi predicts the digit using a **TensorFlow Lite** model.

---

## 1. Project Overview

- **Goal:** Deploy a trained CNN to a Raspberry Pi and perform real-time digit recognition from a camera.
- **Dataset:** MNIST (28×28 grayscale digits 0–9).
- **Model:** Convolutional Neural Network (TensorFlow/Keras).
- **Deployment:** Converted to TensorFlow Lite (`.tflite`) and run on a Raspberry Pi using `tflite-runtime`, `Picamera2`, and OpenCV.

---

## 2. Repository Structure

```text
raspi-digit-recognition/
├── models/
│   └── mnist_cnn.tflite        # Trained TensorFlow Lite model
├── raspi_cam_mnist.py         # Raspberry Pi camera + inference script
└── train_mnist.py             # (Optional) Training script for PC
  -train_mnist.py is used on a PC (Windows/Linux/macOS) to train and export the model.
  -raspi_cam_mnist.py and models/mnist_cnn.tflite are used on the Raspberry Pi.

Requirements:
Hardware
 -Raspberry Pi 4 (recommended)
 -Official Raspberry Pi Camera (CSI interface) or compatible camera
 -MicroSD card with Raspberry Pi OS / Debian Bookworm (64-bit)
 -Monitor, keyboard, mouse for the Pi
 -Optional: Windows / Linux / macOS PC for training
Software on PC (for training)
 -Python 3.10+
 -tensorflow, numpy (and optionally matplotlib for plots)
Software on Raspberry Pi (for inference)
 -All steps below assume:
 -Python 3 is installed (python3 --version)
 -You are using system Python (not in a venv) when running the Pi script

Training the Model on a PC (Optional but Recommended):
If you already have mnist_cnn.tflite, you can skip to Section 5.
1. Clone or download this repo to your PC.

2. Create and activate a virtual environment (recommended):
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

3. Install dependencies:
pip install --upgrade pip
pip install tensorflow numpy

4. Run training script:
python train_mnist.py
 A typical train_mnist.py will:
  -Download MNIST
  -Build and train the CNN
  -Evaluate on the test set
  -Export a TensorFlow Lite model to models/mnist_cnn.tflite

5. Verify the file exists:
ls models
You should see:
  mnist_cnn.tflite

6. Copy mnist_cnn.tflite to the Raspberry Pi:
 Options:
  -USB drive
  -scp over SSH
  -Shared network folder
On the Pi, it should end up in:
~/raspi_digit_recognition/models/mnist_cnn.tflite
(I picked the easier USB Drive method)

Setting Up the Raspberry Pi:
 Enable the Camera:
  1. Open terminal on the Pi.
  2. Run:
      sudo raspi-config
  3. Navigate to:
      Interface Options → Camera → Enable
  4. Reboot the Pi:
      sudo reboot
 Install Required Packages (System Python)
  5. In a terminal on the Pi:
      sudo apt update
      sudo apt install -y python3-opencv python3-picamera2 python3-pip
      sudo pip3 install tflite-runtime numpy --break-system-packages
  6. Check that imports work:
      python3 - << 'EOF'
      import cv2
      from picamera2 import Picamera2
      from tflite_runtime.interpreter import Interpreter
      print("All imports OK")
      EOF
  - If no error appears and “All imports OK” prints, you’re set.
   7. Create Project Folder and Model Directory
        mkdir -p ~/raspi_digit_recognition/models
        cd ~/raspi_digit_recognition
    Copy mnist_cnn.tflite into:
        ~/raspi_digit_recognition/models/mnist_cnn.tflite
    Verify:
        ls models
        # should show:
        # mnist_cnn.tflite
The Inference Script (raspi_cam_mnist.py)
  This script:
    1. Captures frames from the Pi camera using Picamera2
    2. Preprocesses the image (grayscale, blur, threshold, contour, crop, resize to 28×28)
    3. Runs the TensorFlow Lite model (mnist_cnn.tflite)
    4. Displays prediction and confidence, and saves the last prediction image
    5. Place raspi_cam_mnist.py in ~/raspi_digit_recognition/.
  Basic usage:
    cd ~/raspi_digit_recognition
    python3 raspi_cam_mnist.py
  When running, the script:
    - Prints Model loaded. and Camera started.
    - Shows a window with the live camera feed
    - You can press:
        c → capture & classify current frame
        q → quit the program
How to Run the Digit Recognizer (Step by Step)
  1. On the Raspberry Pi, open a terminal.
  2. Make sure the model exists:
      ls ~/raspi_digit_recognition/models
      # mnist_cnn.tflite should be listed
  3. Change into the project directory:
      cd ~/raspi_digit_recognition
  4. Run the script:
      python3 raspi_cam_mnist.py
  5. You should see:
    - In the terminal:
      -model input/output details
      -Camera started.
      -Press 'c' to capture & classify, 'q' to quit.
    - On screen: live camera window
  6. Write a digit (0–9) on white paper with a dark pen/marker.
  7. Hold the paper so the digit fills a good portion of the camera view.
  8. Press c on the keyboard:
    8.1. The code captures a frame
    8.2. Locates the largest digit-like blob
    8.3. Crops and resizes it to 28×28
    8.4. Normalizes and runs the CNN
    8.5. Draws a green box around the digit
    8.6. Shows a second window with the prediction
    8.7. Saves last_prediction.jpg in the project folder
    8.8. Prints something like:
        Predicted: 7, conf=0.98, time=5.2 ms
    8.9.Press q to exit.
  9. How It Works (Brief Summary)
    - Model training (PC): Train a CNN on MNIST and export mnist_cnn.tflite.
    - Deployment (Pi):
      - Load mnist_cnn.tflite using tflite_runtime.Interpreter
      - Grab frames from Picamera2
      - Preprocess frame to match MNIST (28×28, grayscale, normalized)
      - Run inference and get a probability for each digit (0–9)
      - Display the predicted digit and confidence on the frame
This project demonstrates a complete deep-learning pipeline from training to embedded deployment on low-cost hardware.


