"""
Emotion Detection Web Application - Final Version
Uses TensorFlow directly (now working!)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
import joblib
import base64
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
label_encoder = None
emotions = None

# Face detection - use local cascade file
cascade_path = 'haarcascade_frontalface_default.xml'
if os.path.exists(cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print(f"[OK] Haar cascade loaded from: {cascade_path}")
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print(f"[INFO] Using default Haar cascade location")


def load_model():
    """Load the trained model"""
    global model, label_encoder, emotions
    
    try:
        # Load model
        model_path = 'models/emotion_dnn_model.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"[OK] Model loaded from {model_path}")
        else:
            print(f"[ERROR] Model not found: {model_path}")
            return False
        
        # Load label encoder
        encoder_path = 'models/label_encoder.pkl'
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            emotions = label_encoder.classes_
            print(f"[OK] Label encoder loaded: {list(emotions)}")
        else:
            print(f"[ERROR] Label encoder not found: {encoder_path}")
            return False
        
        return True
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_image(image):
    """Preprocess image for model input"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        face_roi = gray
    
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi.astype('float32') / 255.0
    face_roi = face_roi.reshape(1, 48, 48, 1)
    
    return face_roi, image


def predict_emotion(processed_image):
    """Predict emotion from preprocessed image"""
    if model is None:
        return None, None
    
    try:
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        emotion = emotions[predicted_class]
        return emotion, confidence
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        processed_image, annotated_image = preprocess_image(image)
        emotion, confidence = predict_emotion(processed_image)
        
        if emotion is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        predictions = model.predict(processed_image, verbose=0)[0]
        all_emotions = {emotions[i]: float(predictions[i]) for i in range(len(emotions))}
        
        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': confidence,
            'image': img_base64,
            'all_predictions': all_emotions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video'}), 400
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_skip = max(1, fps // 5)
        
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                processed_image, annotated_frame = preprocess_image(frame)
                emotion, confidence = predict_emotion(processed_image)
                
                if emotion:
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    results.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'emotion': emotion,
                        'confidence': confidence,
                        'image': frame_base64
                    })
            
            frame_count += 1
            if len(results) >= 100:
                break
        
        cap.release()
        os.remove(filepath)
        
        if not results:
            return jsonify({'error': 'No faces detected'}), 400
        
        emotions_detected = [r['emotion'] for r in results]
        
        return jsonify({
            'success': True,
            'processed_frames': len(results),
            'fps': fps,
            'results': results,
            'emotions_detected': list(set(emotions_detected)),
            'dominant_emotion': max(set(emotions_detected), key=emotions_detected.count)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_webcam_frames():
    """Generator for webcam streaming"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    emotion_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 3 == 0:
            processed_image, annotated_frame = preprocess_image(frame)
            emotion, confidence = predict_emotion(processed_image)
            
            if emotion:
                emotion_history.append(emotion)
                if len(emotion_history) > 10:
                    emotion_history.pop(0)
                
                dominant_emotion = max(set(emotion_history), key=emotion_history.count)
                cv2.putText(annotated_frame, f"Emotion: {dominant_emotion}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Confidence: {confidence*100:.1f}%", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_count += 1
    
    cap.release()


@app.route('/webcam')
def webcam():
    return Response(generate_webcam_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'emotions': list(emotions) if emotions is not None else []
    })


if __name__ == '__main__':
    print("="*60)
    print("EMOTION DETECTION - TENSORFLOW VERSION")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    
    print("\nLoading model...")
    if not load_model():
        print("\n[ERROR] Failed to load model")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("STARTING WEB SERVER")
    print("="*60)
    print("\nOpen browser: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)