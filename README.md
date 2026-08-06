# Emotion Detection Web Application

## Installation & Setup

### Step 1: Install Visual C++ Redistributable
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
python emotion_detection_app.py
```

Then open your browser and navigate to: http://localhost:5000

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