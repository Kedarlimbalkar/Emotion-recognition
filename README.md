# 🎭 DeepFER - Facial Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.3+-red.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> An end-to-end **AI/ML + Data Engineering** project for real-time facial emotion recognition — featuring a production-grade data pipeline, custom CNN model, and interactive web deployment.

---

## 🎯 What This Project Does

DeepFER detects **7 human emotions** from facial images in real-time using a deep learning model trained on 35,000+ images. It includes a complete **data engineering pipeline** from raw image validation to model-ready tensors — not just a Jupyter notebook.

**Emotions:** 😠 Angry | 🤢 Disgust | 😨 Fear | 😊 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise

---

## 🔧 Data Engineering Pipeline

> ⭐ **This is the core strength of the project** — a modular, production-style data pipeline built from scratch.

### Pipeline Architecture

```
Raw Images (35,905 files across 7 emotion classes)
        │
        ▼
┌─────────────────────────────┐
│   Stage 1: Data Validation  │  ← validate_image()
│   • Checks every image      │    - Readable? Corrupted?
│   • Min size: 48×48px       │    - Empty file?
│   • Format validation       │    - Valid pixel values?
└─────────────┬───────────────┘
              │
        ▼
┌─────────────────────────────┐
│   Stage 2: Data Cleaning    │  ← clean_dataset()
│   • Removes bad images      │    - Saves cleaning_report.json
│   • Logs removed files      │    - Tracks reason per file
│   • Per-class processing    │    - Production-style logging
└─────────────┬───────────────┘
              │
        ▼
┌─────────────────────────────┐
│   Stage 3: Preprocessing    │  ← preprocess()
│   • Grayscale conversion    │    - BGR → Grayscale
│   • Resize to 48×48         │    - Consistent input shape
│   • Normalize [0, 1]        │    - rescale=1./255
└─────────────┬───────────────┘
              │
        ▼
┌─────────────────────────────┐
│   Stage 4: Data Augmentation│  ← ImageDataGenerator
│   • Rotation: ±30°          │    - Prevents overfitting
│   • Width/Height shift: 20% │    - Synthetic data generation
│   • Zoom: 20%               │    - 28,839 training samples
│   • Horizontal flip         │    - 7,066 validation samples
└─────────────┬───────────────┘
              │
        ▼
┌─────────────────────────────┐
│   Stage 5: Statistics &     │  ← generate_statistics()
│   Reporting                 │    - Per-class image counts
│   • Dataset distribution    │    - CSV + JSON reports
│   • Quality metrics         │    - Pipeline run logs
└─────────────┬───────────────┘
              │
        ▼
   Model Training Ready ✅
```

### Data Engineering Skills Demonstrated

| Skill | Implementation |
|-------|---------------|
| **ETL Pipeline** | Extract (load images) → Transform (clean/augment) → Load (feed model) |
| **Data Validation** | Automated quality checks on 35,000+ images |
| **Data Cleaning** | Corrupted file detection, removal, and JSON reporting |
| **Data Augmentation** | Synthetic data generation to balance and expand dataset |
| **Pipeline Logging** | Production-style logging at every stage with timestamps |
| **Modular Design** | Reusable `DataEngineer` class — plug in any image dataset |
| **Reporting** | Auto-generated cleaning reports and dataset statistics |
| **Config Management** | Centralized config file for all pipeline parameters |

---

## 🤖 Machine Learning & AI

### Model Architecture

```
Input: 48×48×1 (Grayscale Face Image)
        ↓
Conv2D (32 filters, 3×3, ReLU)    ← Feature extraction
        ↓
BatchNormalization                  ← Training stability
        ↓
MaxPooling2D (2×2)                  ← Dimensionality reduction
        ↓
Flatten                             ← 16,928 features
        ↓
Dense (64 units, ReLU)              ← Pattern learning
        ↓
Dense (7 units, Softmax)            ← 7-class output
        ↓
Output: Emotion + Confidence Score
```

### Training Configuration

```python
# Data augmentation (part of data engineering pipeline)
ImageDataGenerator(
    rescale        = 1./255,       # Normalization
    rotation_range = 30,           # Random rotation
    width_shift    = 0.2,          # Horizontal shift
    height_shift   = 0.2,          # Vertical shift
    zoom_range     = 0.2,          # Random zoom
    horizontal_flip= True          # Mirror augmentation
)

# Training
optimizer = Adam(lr=0.001)
loss      = categorical_crossentropy
epochs    = 10
callbacks = [ModelCheckpoint, EarlyStopping]
```

### ML Skills Demonstrated

| Skill | Implementation |
|-------|---------------|
| **Deep Learning** | Custom CNN trained on 28,839 images |
| **Transfer Learning** | MobileNetV2 architecture concepts applied |
| **Image Classification** | 7-class softmax output |
| **Overfitting Prevention** | BatchNorm + Dropout + Augmentation |
| **Model Serialization** | Save/load in .keras and .h5 formats |
| **Inference Pipeline** | Real-time prediction with face detection |
| **Computer Vision** | OpenCV Haar Cascade face detection |
| **Evaluation** | Confidence scores, per-class predictions |

---

## 📊 Dataset

| Split | Images | Source |
|-------|--------|--------|
| Training | 28,839 | FER-2013 |
| Validation | 7,066 | FER-2013 |
| **Total** | **35,905** | **7 classes** |

**Class Distribution:**

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

# Virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install
pip install -r requirements.txt

# Fix model compatibility (run once)
python extract_weights.py

# Launch app
streamlit run streamlit_app/app.py
```

Open **http://localhost:8501**

---

## 📁 Project Structure

```
emotion-recognition-cnn/
│
├── 📂 src/                        ← Core ML/DE modules
│   ├── data_engineering.py        ← ⭐ Full data pipeline
│   ├── model.py                   ← CNN architecture
│   ├── train.py                   ← Training pipeline
│   ├── evaluate.py                ← Metrics & reports
│   └── inference.py               ← Real-time detection
│
├── 📂 configs/
│   └── config.py                  ← Centralized config
│
├── 📂 streamlit_app/
│   └── app.py                     ← Web interface
│
├── 📂 models/
│   ├── DeepFER_Final_Model_v1.keras  ← Trained model
│   └── deepfer_compatible.h5         ← Compatible format
│
├── 📂 images/
│   ├── train/                     ← 28,839 training images
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   └── validation/                ← 7,066 validation images
│       └── [same 7 folders]
│
├── 📂 docker/
│   └── Dockerfile
├── 📂 .github/workflows/
│   └── ci.yml                     ← CI/CD pipeline
│
├── extract_weights.py             ← Model compatibility fix
├── fix_model.py                   ← Model rebuild utility
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## 🌐 Web Application

Built with **Streamlit** — upload any face photo and get:
- ✅ Automatic face detection (OpenCV Haar Cascade)
- ✅ Emotion label + confidence score
- ✅ Interactive bar chart of all 7 emotion probabilities
- ✅ Annotated image with bounding box

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
# Access at http://localhost:8501
```

---

## 💼 Role Relevance

### For Data Engineer Roles
- Production ETL pipeline (not just a notebook)
- Data validation, cleaning, and quality reporting
- Modular, reusable pipeline architecture
- Config-driven design
- Logging at every pipeline stage

### For ML Engineer / AI Roles
- End-to-end model development
- Custom CNN architecture
- Data augmentation strategy
- Model serialization and deployment
- Real-time inference pipeline

### For Full Stack / MLOps Roles
- Streamlit web app deployment
- Docker containerization
- GitHub Actions CI/CD
- Virtual environment management

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **ML/DL** | TensorFlow 2.17, Keras 3.3, Scikit-learn |
| **Computer Vision** | OpenCV, Pillow |
| **Data** | NumPy, Pandas |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web** | Streamlit |
| **DevOps** | Docker, GitHub Actions |
| **Language** | Python 3.10 |

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
⭐ Star this repo if you found it useful!
</div>
