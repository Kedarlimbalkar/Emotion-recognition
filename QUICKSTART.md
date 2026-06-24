# 🚀 Quick Start Guide

Get DeepFER running in 5 minutes!

## 📋 Prerequisites

- Python 3.11+
- Git
- 8GB RAM (4GB minimum)
- Optional: NVIDIA GPU with CUDA

## ⚡ 5-Minute Setup

### Step 1: Clone Repository (30 seconds)
```bash
git clone https://github.com/Kedarlimbalkar/Emotion-recognition.git
cd emotion-recognition-cnn
```

### Step 2: Create Virtual Environment (1 minute)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies (2 minutes)
```bash
pip install -r requirements.txt
```

### Step 4: Run Web App (1 minute)
```bash
streamlit run streamlit_app/app.py
```

**Your app is now running at: http://localhost:8501** 🎉

## 📸 Using the Web App

### Image Detection
1. Go to **Image Detection** tab
2. Upload a photo containing faces
3. See real-time emotion predictions!

### Video Detection
1. Go to **Video Detection** tab
2. Upload a video or use webcam
3. Watch emotion detection in action!

## 💻 Command Line Usage

### Test on Single Image
```python
python -c "
from src.inference import EmotionDetector
detector = EmotionDetector('models/DeepFER_Final_Model.keras')
results, img = detector.detect_and_recognize_emotions('test.jpg')
for r in results:
    print(f'{r[\"emotion\"]}: {r[\"confidence\"]:.2%}')
"
```

### Test on Video
```python
python -c "
from src.inference import EmotionDetector
detector = EmotionDetector('models/DeepFER_Final_Model.keras')
detector.detect_emotions_video('test.mp4', 'output.mp4')
"
```

### Batch Process Directory
```python
python -c "
from src.inference import BatchPredictor
predictor = BatchPredictor('models/DeepFER_Final_Model.keras')
results = predictor.predict_directory('images/', 'output/')
"
```

## 🐳 Docker Setup (Alternative)

### One-Command Deployment
```bash
docker-compose up
```

Access at: http://localhost:8501

## 📂 Folder Structure
```
emotion-recognition-cnn/
├── streamlit_app/app.py      # Web interface
├── src/
│   ├── inference.py          # For predictions
│   ├── train.py              # For training
│   └── evaluate.py           # For evaluation
├── models/                   # Pre-trained models
├── data/                     # Your data here
└── README.md                 # Full documentation
```

## 🎯 Common Tasks

### Train on Your Own Data

```bash
# 1. Organize data in folder structure:
# data/raw/
#   ├── Angry/
#   ├── Happy/
#   ├── Sad/
#   └── ...

# 2. Prepare data
python prepare_data.py --source data/raw --output data/processed

# 3. Train model
python src/train.py

# 4. Evaluate
python src/evaluate.py
```

### Run on Webcam

```python
from src.inference import EmotionDetector
detector = EmotionDetector()
detector.detect_emotions_video()  # Webcam feed
```

### Get Detailed Predictions

```python
from src.inference import EmotionDetector
detector = EmotionDetector()
results, img = detector.detect_and_recognize_emotions('photo.jpg')

for face in results:
    print(f"Emotion: {face['emotion']}")
    print(f"Confidence: {face['confidence']:.2%}")
    print(f"All predictions: {face['all_predictions']}")
```

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Fix:** Install requirements: `pip install -r requirements.txt`

### Issue: "Port 8501 already in use"
**Fix:** Use different port: `streamlit run streamlit_app/app.py --server.port 8502`

### Issue: "Model file not found"
**Fix:** Download from GitHub releases and place in `models/` folder

### Issue: "CUDA not available"
**Fix:** This is fine! TensorFlow will use CPU. For GPU, install:
```bash
pip install tensorflow[and-cuda]
```

### Issue: "Out of memory"
**Fix:** Reduce batch size in `configs/config.py` (line ~19)

## 📚 Next Steps

1. **Read Full Documentation**: See [README.md](README.md)
2. **Explore Jupyter Notebooks**: Check `notebooks/` folder
3. **Train Custom Model**: Use your own dataset
4. **Deploy to Cloud**: Push to Heroku/AWS/GCP
5. **Contribute**: Submit PRs for improvements

## 🎓 Learning Resources

- [TensorFlow Documentation](https://tensorflow.org)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Streamlit Tutorial](https://docs.streamlit.io)
- [FER2013 Dataset Paper](https://arxiv.org/abs/1307.0414)

## 💬 Getting Help

- 🐛 **Found a bug?** Open an issue on GitHub
- 💡 **Have an idea?** Create a discussion
- 📧 **Need help?** Email: kedar@example.com

## 🌟 Support

If this project helped you, please:
- ⭐ Star the repository
- 🔗 Share with friends
- 🤝 Contribute improvements
- 📣 Spread the word!

---

**Happy emotion detecting! 😊**
