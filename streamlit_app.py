"""
Emotion Detection Web Application - Streamlit Version
Deploy easily with: streamlit run streamlit_app.py
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import joblib
import tempfile

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        color: #667eea;
        margin-bottom: 30px;
    }
    .emotion-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin: 20px 0;
    }
    .confidence-bar {
        background: #e0e0e0;
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'emotions' not in st.session_state:
    st.session_state.emotions = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Load model
        model_path = 'models/emotion_dnn_model.keras'
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found: {model_path}")
            st.info("Please ensure the model file exists in the models/ directory")
            return None, None
        
        # Show loading message
        with st.spinner("Loading model... This may take a minute."):
            model = tf.keras.models.load_model(model_path)
            st.success(f"✅ Model loaded successfully!")
        
        # Load label encoder
        encoder_path = 'models/label_encoder.pkl'
        if not os.path.exists(encoder_path):
            st.error(f"❌ Label encoder not found: {encoder_path}")
            return None, None
        
        label_encoder = joblib.load(encoder_path)
        emotions = label_encoder.classes_
        st.success(f"✅ Emotions detected: {list(emotions)}")
        
        return model, emotions
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try to detect face
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                face_roi = gray
        except:
            face_roi = gray
        
        # Resize and normalize
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = face_roi.reshape(1, 48, 48, 1)
        
        return face_roi, image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def predict_emotion(processed_image):
    """Predict emotion from preprocessed image"""
    if st.session_state.model is None:
        return None, None
    
    try:
        predictions = st.session_state.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        emotion = st.session_state.emotions[predicted_class]
        return emotion, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def main():
    # Header
    st.markdown('<div class="main-header">🎭 Emotion Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ About")
        st.info("""
        **Real-Time Emotion Detection**
        
        This app uses a Deep Neural Network to detect emotions from:
        - 📷 Images
        - 🎬 Videos  
        - 📹 Webcam (Live)
        
        **Emotions detected:**
        - Angry
        - Disgust
        - Fear
        - Happy
        - Neutral
        - Sad
        - Surprise
        """)
        
        st.subheader("📊 Model Status")
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded")
            if st.session_state.emotions is not None:
                st.write(f"**Emotions:** {list(st.session_state.emotions)}")
        else:
            st.warning("⚠️ Model not loaded yet")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading model... Please wait..."):
            model, emotions = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.emotions = emotions
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                st.rerun()
    
    # Main content
    if st.session_state.model_loaded:
        st.success("✅ Model is ready! Upload an image to start.")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎬 Video Upload", "📹 Webcam"])
        
        # Tab 1: Image Upload
        with tab1:
            st.header("Upload Image for Emotion Detection")
            
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image containing a face"
            )
            
            if uploaded_file is not None:
                try:
                    # Read image
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        # Display original image
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original Image")
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                        
                        # Process and predict
                        with st.spinner("Analyzing emotion..."):
                            processed_image, annotated_image = preprocess_image(image)
                            
                            if processed_image is not None:
                                emotion, confidence = predict_emotion(processed_image)
                                
                                with col2:
                                    st.subheader("Result")
                                    if emotion:
                                        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                                        
                                        # Display emotion
                                        st.markdown(f'<div class="emotion-box">{emotion.upper()}</div>', unsafe_allow_html=True)
                                        
                                        # Display confidence
                                        confidence_pct = confidence * 100
                                        st.markdown(f"""
                                        <div class="confidence-bar">
                                            <div class="confidence-fill" style="width: {confidence_pct}%">
                                                {confidence_pct:.1f}%
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Show all predictions
                                        with st.expander("📊 All Predictions"):
                                            predictions = st.session_state.model.predict(processed_image, verbose=0)[0]
                                            for i, emo in enumerate(st.session_state.emotions):
                                                prob = predictions[i] * 100
                                                st.progress(prob / 100)
                                                st.write(f"{emo}: {prob:.1f}%")
                                    else:
                                        st.error("❌ Could not detect emotion")
                            else:
                                st.error("❌ Error processing image")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Tab 2: Video Upload
        with tab2:
            st.header("Upload Video for Emotion Analysis")
            
            uploaded_video = st.file_uploader(
                "Choose a video...",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for emotion analysis"
            )
            
            if uploaded_video is not None:
                try:
                    # Save video temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_video.read())
                        video_path = tmp_file.name
                    
                    # Process video
                    cap = cv2.VideoCapture(video_path)
                    
                    if cap.isOpened():
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_skip = max(1, fps // 5)
                        
                        st.info(f"📹 Video: {fps} FPS, {total_frames} total frames")
                        
                        results = []
                        frame_count = 0
                        progress_bar = st.progress(0)
                        
                        with st.spinner("Processing video..."):
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                if frame_count % frame_skip == 0:
                                    processed_image, annotated_frame = preprocess_image(frame)
                                    emotion, confidence = predict_emotion(processed_image)
                                    
                                    if emotion:
                                        results.append({
                                            'frame': frame_count,
                                            'time': frame_count / fps,
                                            'emotion': emotion,
                                            'confidence': confidence
                                        })
                                
                                frame_count += 1
                                progress_bar.progress(min(frame_count / total_frames, 1.0))
                                
                                if len(results) >= 50:
                                    break
                        
                        cap.release()
                        os.remove(video_path)
                        
                        if results:
                            # Statistics
                            st.subheader("📊 Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            emotions_detected = [r['emotion'] for r in results]
                            unique_emotions = list(set(emotions_detected))
                            dominant_emotion = max(set(emotions_detected), key=emotions_detected.count)
                            
                            with col1:
                                st.metric("Frames Analyzed", len(results))
                            with col2:
                                st.metric("FPS", fps)
                            with col3:
                                st.metric("Dominant Emotion", dominant_emotion)
                            with col4:
                                st.metric("Emotions Found", len(unique_emotions))
                            
                            # Sample frames
                            st.subheader("🎬 Sample Frames")
                            for i in range(0, min(len(results), 10), 2):
                                cols = st.columns(2)
                                for j in range(2):
                                    if i + j < len(results):
                                        result = results[i + j]
                                        with cols[j]:
                                            st.write(f"**{result['emotion']}** ({result['confidence']*100:.1f}%)")
                                            st.write(f"Time: {result['time']:.1f}s")
                        else:
                            st.warning("No faces detected in video")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Tab 3: Webcam
        with tab3:
            st.header("Real-time Webcam Emotion Detection")
            st.warning("⚠️ Webcam feature works best when running locally. On Streamlit Cloud, use Image Upload instead.")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Webcam placeholder
                video_placeholder = st.empty()
            
            with col2:
                st.subheader("Controls")
                start_button = st.button("▶️ Start Webcam")
                stop_button = st.button("⏹️ Stop Webcam")
                
                if start_button:
                    st.session_state.webcam_running = True
                if stop_button:
                    st.session_state.webcam_running = False
            
            # Webcam streaming
            if st.session_state.get('webcam_running', False):
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                frame_count = 0
                emotion_history = []
                
                while st.session_state.get('webcam_running', False):
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
                    
                    # Display frame
                    video_placeholder.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_column_width=True
                    )
                    
                    frame_count += 1
                
                cap.release()
                st.session_state.webcam_running = False
            else:
                st.info("👆 Click 'Start Webcam' to begin real-time emotion detection")
    else:
        st.error("❌ Model not loaded. Please check the error messages above.")
        st.info("💡 Tip: Make sure the model files exist in the models/ directory")

if __name__ == '__main__':
    main()