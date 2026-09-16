# Real-Time Age and Gender Detection using OpenCV & Deep Learning

A computer vision application built with Python and OpenCV DNN to detect faces in real-time or from static images and accurately predict demographic details (Age Range and Gender).

---

## 📌 Project Overview

This project utilizes pre-trained Deep Neural Network (DNN) Caffe and TensorFlow models to perform three-stage processing:
1. **Face Detection**: Uses an SSD (Single Shot MultiBox Detector) model to locate human faces in frames.
2. **Gender Classification**: Predicts whether a face belongs to `Male` or `Female`.
3. **Age Estimation**: Classifies detected faces into one of 8 distinct age brackets: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`.

---

## 🛠 Tech Stack & Tools

* **Programming Language:** Python 3.8+
* **Core Libraries:** 
  * `opencv-python`: Image and video frame processing + DNN model inference.
  * `numpy`: Array manipulations and matrix operations.
  * `argparse`: Command-line interface argument parsing.
* **Model Frameworks:** Caffe Models (`.prototxt`, `.caffemodel`) / TensorFlow (`.pb`, `.pbtxt`).

---

## 📂 Project Structure

```text
├── models/
│   ├── opencv_face_detector.pbtxt       # Face detector architecture
│   ├── opencv_face_detector_uint8.pb    # Face detector trained weights
│   ├── deploy_gender.prototxt           # Gender model architecture
│   ├── gender_net.caffemodel            # Gender model trained weights
│   ├── deploy_age.prototxt              # Age model architecture
│   └── age_net.caffemodel               # Age model trained weights
├── images/                              # Sample test images
├── detect.py                            # Main application script
├── requirements.txt                     # Project dependencies
└── README.md                            # Documentation
