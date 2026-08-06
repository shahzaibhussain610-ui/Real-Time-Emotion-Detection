# Real-Time Emotion Detection

A deep learning-based web application for real-time emotion detection using a trained DNN model. The application can detect 7 different emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Features

- 📷 **Image Upload** - Upload images for emotion detection with face detection
- 🎬 **Video Upload** - Upload videos for frame-by-frame emotion analysis
- 📹 **Webcam** - Real-time emotion detection from webcam
- 🎨 **Modern UI** - Beautiful gradient design with drag-and-drop support
- 📊 **Confidence Scores** - Shows prediction confidence for all emotions

## Live Demo

Access the application at: **http://localhost:5000** (when running locally)

## Repository

GitHub: https://github.com/shahzaibhussain610-ui/Real-Time-Emotion-Detection

## Installation & Setup

### Option 1: Run with Flask (Original)
#### Step 1: Install Visual C++ Redistributable
The error you're seeing is due to missing Visual C++ Redistributable. Download and install it:

**Download Link:** https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart your computer.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Alternative - Use TensorFlow CPU Version (if above fails)
If you still face issues, uninstall TensorFlow and install the CPU-only version:
```bash
pip uninstall tensorflow keras
pip install tensorflow-cpu==2.15.0
```

### Step 4: Run the Application
```bash
python emotion_detection_final.py
```

Then open your browser and navigate to: http://localhost:5000

---

### Option 2: Run with Streamlit (Recommended for Deployment)
#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at: http://localhost:8501

#### Deploy to Streamlit Cloud
1. Push your code to GitHub (already done!)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Select your repository: `shahzaibhussain610-ui/Real-Time-Emotion-Detection`
5. Set the main file path: `streamlit_app.py`
6. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

## Features
- 📷 **Image Upload**: Upload images for emotion detection
- 🎬 **Video Upload**: Upload videos for frame-by-frame analysis
- 📹 **Webcam**: Real-time emotion detection from webcam

## Troubleshooting

### TensorFlow DLL Error
If you get "Failed to load _pywrap_tensorflow_common.dll":
1. Install Visual C++ Redistributable (link above)
2. Restart your computer
3. If still failing, use tensorflow-cpu instead

### Webcam Not Working
- Ensure your webcam is connected and not being used by another application
- Grant camera permissions to your browser

### Model Not Found
Ensure these files exist:
- `models/emotion_dnn_model.keras`
- `models/label_encoder.pkl`

## Project Structure
```
Day5/
├── emotion_detection_app.py  # Main Flask application
├── templates/
│   └── index.html            # Web UI
├── models/
│   ├── emotion_dnn_model.keras
│   └── label_encoder.pkl
├── uploads/                  # Temporary upload folder
└── requirements.txt