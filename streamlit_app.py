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
        - 📹 Webcam
        
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
        tab1, tab2 = st.tabs(["📷 Image Upload", "📹 Webcam"])
        
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
        
        # Tab 2: Webcam
        with tab2:
            st.header("Real-time Webcam Emotion Detection")
            st.warning("⚠️ Webcam feature works best when running locally. On Streamlit Cloud, use Image Upload instead.")
            
            # Webcam input
            webcam_image = st.camera_input("Take a photo with your webcam")
            
            if webcam_image is not None:
                try:
                    # Convert to OpenCV format
                    image = np.array(Image.open(webcam_image))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Process and predict
                    with st.spinner("Analyzing emotion..."):
                        processed_image, annotated_image = preprocess_image(image)
                        
                        if processed_image is not None:
                            emotion, confidence = predict_emotion(processed_image)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Original")
                                st.image(image, channels="BGR", use_column_width=True)
                            
                            with col2:
                                st.subheader("Result")
                                if emotion:
                                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                                    st.markdown(f'<div class="emotion-box">{emotion.upper()}</div>', unsafe_allow_html=True)
                                    st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                                else:
                                    st.error("❌ Could not detect emotion")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    else:
        st.error("❌ Model not loaded. Please check the error messages above.")
        st.info("💡 Tip: Make sure the model files exist in the models/ directory")

if __name__ == '__main__':
    main()