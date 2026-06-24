# 🎭 DeepFER - Facial Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.3+-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end **AI/ML project** for real-time facial emotion recognition — covering data engineering, model development, web deployment, and DevOps.

---

## 🎯 Overview

DeepFER detects **7 human emotions** from facial images using a deep learning CNN model trained on 35,905 images. This project demonstrates a complete production-ready pipeline — from raw data to a live web application.

**Emotions Detected:**
😠 Angry | 🤢 Disgust | 😨 Fear | 😊 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise

---

## 🏗️ Project Architecture

```
Raw Images (35,905)
      ↓
Data Engineering Pipeline
(Validation → Cleaning → Preprocessing → Augmentation)
      ↓
CNN Model Training
(Conv2D → BatchNorm → MaxPool → Dense → Softmax)
      ↓
Model Evaluation & Serialization
      ↓
Streamlit Web App → Docker → GitHub Actions CI/CD
```

---

## ✨ Key Features

- 🔍 **Data Pipeline** — automated image validation, cleaning, and augmentation
- 🧠 **Deep Learning** — custom CNN trained on 35,000+ facial images
- 👁️ **Computer Vision** — real-time face detection using OpenCV
- 🌐 **Web App** — interactive Streamlit interface with live predictions
- 🐳 **Docker** — containerized for easy deployment anywhere
- 🔄 **CI/CD** — GitHub Actions pipeline for automated testing and builds
- 📊 **Analytics** — confidence scores and per-emotion probability charts
- 🧩 **Modular Code** — reusable classes with config-driven design

---

## 🤖 Model Architecture

```
Input: 48×48×1 (Grayscale)
        ↓
Conv2D (32 filters, 3×3, ReLU)
        ↓
BatchNormalization
        ↓
MaxPooling2D (2×2)
        ↓
Flatten → Dense(64, ReLU)
        ↓
Dense(7, Softmax)
        ↓
Emotion + Confidence Score
```

| Parameter | Value |
|-----------|-------|
| Input Shape | 48 × 48 × 1 |
| Total Parameters | ~1.08M |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Output Classes | 7 |

---

## 🔧 Data Engineering Pipeline

```
Raw Images
    ↓  validate_image()      → checks corruption, size, readability
    ↓  clean_dataset()       → removes bad files, saves cleaning_report.json
    ↓  preprocess()          → grayscale, resize 48×48, normalize [0,1]
    ↓  ImageDataGenerator()  → rotation, zoom, flip augmentation
    ↓  generate_statistics() → per-class counts, CSV/JSON reports
    ↓
Model-Ready Tensors ✅
```

**Pipeline handles:** 28,839 training images + 7,066 validation images across 7 classes.

---

## 📊 Dataset

| Split | Images |
|-------|--------|
| Training | 28,839 |
| Validation | 7,066 |
| **Total** | **35,905** |

**Source:** FER-2013 (Facial Expression Recognition)

| Emotion | Train | Validation |
|---------|-------|------------|
| Angry | ~3,993 | ~960 |
| Disgust | ~436 | ~111 |
| Fear | ~4,103 | ~1,018 |
| Happy | ~7,164 | ~1,825 |
| Neutral | ~4,982 | ~1,216 |
| Sad | ~4,938 | ~1,139 |
| Surprise | ~3,205 | ~797 |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Kedarlimbalkar/Emotion-recognition.git
cd Emotion-recognition

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Fix model (run once)
python extract_weights.py

# Launch app
streamlit run streamlit_app/app.py
# Open http://localhost:8501
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
# Access at http://localhost:8501
```

---

## 📁 Project Structure

```
emotion-recognition-cnn/
├── src/
│   ├── data_engineering.py   ← Data pipeline
│   ├── model.py              ← CNN architecture
│   ├── train.py              ← Training pipeline
│   ├── evaluate.py           ← Metrics & reports
│   └── inference.py          ← Real-time detection
├── configs/
│   └── config.py             ← Centralized config
├── streamlit_app/
│   └── app.py                ← Web interface
├── models/
│   └── deepfer_compatible.h5 ← Trained model
├── images/
│   ├── train/                ← 7 emotion folders
│   └── validation/           ← 7 emotion folders
├── docker/Dockerfile
├── .github/workflows/ci.yml  ← CI/CD
├── extract_weights.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | TensorFlow 2.17, Keras 3.3 |
| **Computer Vision** | OpenCV, Pillow |
| **Data Engineering** | NumPy, Pandas, ImageDataGenerator |
| **ML Utilities** | Scikit-learn |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web App** | Streamlit |
| **DevOps** | Docker, GitHub Actions |
| **Language** | Python 3.10 |

---

## 💼 Skills Demonstrated

| Domain | Skills |
|--------|--------|
| **Data Engineering** | ETL pipeline, data validation, cleaning, augmentation, reporting |
| **Machine Learning** | CNN design, training, evaluation, serialization |
| **AI / Deep Learning** | Image classification, feature extraction, softmax output |
| **Computer Vision** | Face detection, image preprocessing, real-time inference |
| **Data Analysis** | Dataset statistics, class distribution, quality metrics |
| **Software Engineering** | OOP, modular design, config management, logging |
| **Web Development** | Streamlit app, interactive charts, file upload |
| **DevOps / MLOps** | Docker, CI/CD with GitHub Actions, virtual environments |
| **Python** | NumPy, Pandas, TensorFlow, OpenCV, Streamlit |

---

## 🌐 Web App Features

- 📷 Upload any face image (JPG/PNG)
- 🎯 Auto face detection with bounding boxes
- 📊 Interactive probability chart for all 7 emotions
- 🖼️ Annotated output image with emotion label
- ℹ️ Model info and dataset details page

---

## 👤 Author

**Kedar Limbalkar**
- GitHub: [@Kedarlimbalkar](https://github.com/Kedarlimbalkar)
- LinkedIn: [linkedin.com/in/kedarlimbalkar](https://linkedin.com/in/kedarlimbalkar)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
⭐ If you found this useful, please give it a star!
</div>
