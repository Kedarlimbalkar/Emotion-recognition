import os
import gdown

def download_model():
    os.makedirs('models', exist_ok=True)
    if not os.path.exists('models/deepfer_compatible.h5'):
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=13hydmdPuK3L7Z9ccS0XQdciItHcn_Yud"
        gdown.download(url, 'models/deepfer_compatible.h5', quiet=False)
        print("Model downloaded successfully!")
    else:
        print("Model already exists!")

download_model()