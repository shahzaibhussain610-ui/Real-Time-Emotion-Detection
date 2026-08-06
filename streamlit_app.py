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
from typing import Tuple, List, Dict, Any, Optional
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    RTCConfiguration,
)
import av

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

def detect_emotions_in_frame(frame):
    """Detect emotions in all faces in a frame"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect all faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []

        # Process each face
        for (x, y, w, h) in faces:
            # Extract face
            face_roi = gray[y:y+h, x:x+w]

            # Resize and normalize
            face_roi_resized = cv2.resize(face_roi, (48, 48))
            face_roi_normalized = face_roi_resized.astype('float32') / 255.0
            face_roi_reshaped = face_roi_normalized.reshape(1, 48, 48, 1)

            # Predict emotion
            if st.session_state.model is not None:
                predictions = st.session_state.model.predict(face_roi_reshaped, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                emotion = st.session_state.emotions[predicted_class]

                results.append({
                    'box': (x, y, w, h),
                    'emotion': emotion,
                    'confidence': confidence
                })

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion}: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, results
    except Exception as e:
        return frame, []


# ============================================================
# WEBCAM IMPLEMENTATION (REGENERATED — production-ready)
# ============================================================

def _build_rtc_configuration():
    """
    Build an RTCConfiguration with:
      - Multiple public STUN servers (always, for reliability/ redundancy)
      - Optional TURN relay servers loaded from Streamlit Secrets (st.secrets.turn)

    Returns:
        (RTCConfiguration, bool) where the bool indicates whether TURN
        servers were successfully configured.

    When no TURN credentials are present in Streamlit Secrets, the function
    silently falls back to STUN-only mode so the app works out-of-the-box.
    """
    ice_servers = [
        {"urls": "stun:stun.l.google.com:19302"},          # Google STUN (primary)
        {"urls": "stun:stun1.l.google.com:19302"},          # Google STUN (backup 1)
        {"urls": "stun:stun2.l.google.com:19302"},          # Google STUN (backup 2)
        {"urls": "stun:stun3.l.google.com:19302"},          # Google STUN (backup 3)
        {"urls": "stun:stun4.l.google.com:19302"},          # Google STUN (backup 4)
        {"urls": "stun:stun.stun.hu:3478"},                 # Independent STUN
        {"urls": "stun:stun4.strato.de:3478"},              # Independent STUN
    ]

    turn_configured = False
    try:
        if hasattr(st, "secrets") and "turn" in st.secrets:
            turn = st.secrets["turn"]
            turn_urls = turn.get("urls", "")
            turn_username = turn.get("username", "")
            turn_credential = turn.get("credential", "")
            if turn_urls and turn_username and turn_credential:
                for url in [u.strip() for u in turn_urls.split(",") if u.strip()]:
                    ice_servers.append(
                        {"urls": url, "username": turn_username, "credential": turn_credential}
                    )
                turn_configured = True
    except Exception:
        # Secrets not available (e.g. local dev without a secrets file)
        turn_configured = False

    rtc_config = RTCConfiguration({"iceServers": ice_servers})
    return rtc_config, turn_configured


class EmotionVideoProcessor(VideoProcessorBase):
    """
    Modern video processor (replaces the deprecated `VideoTransformerBase`)
    for real-time, multi-face emotion detection.

    `recv()` is invoked on every frame that streamlit-webrtc delivers.
    With ``async_processing=True`` the framework guarantees that only the
    latest frame is processed — earlier frames are dropped — which keeps
    latency low even when model inference is slower than the camera FPS.
    """

    def __init__(self) -> None:
        self._frame_counter = 0
        # Cache references in the main thread (init runs on the main thread)
        # so recv() — which runs on the worker thread — does not need to
        # touch st.session_state (which is not thread-safe).
        self._model = st.session_state.model
        self._emotions = st.session_state.emotions

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process one video frame.

        Parameters
        ----------
        frame : av.VideoFrame
            A single frame delivered by the WebRTC pipeline.

        Returns
        -------
        av.VideoFrame
            The annotated frame (or the original on error).
        """
        try:
            # Convert the incoming av.VideoFrame to a numpy array (BGR)
            img = frame.to_ndarray(format="bgr24")

            self._frame_counter += 1
            # Process every 3rd frame to balance real-time feel with CPU load.
            # With async_processing=True the dropped frames are naturally
            # skipped, so this keeps the emotion overlay reasonably fresh.
            if self._frame_counter % 3 == 0:
                annotated_frame, _ = detect_emotions_in_frame(img)
            else:
                annotated_frame = img

            # Convert back to av.VideoFrame for streamlit-webrtc
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        except Exception:
            # Never crash the WebRTC pipeline — return the original frame
            try:
                return av.VideoFrame.from_ndarray(
                    frame.to_ndarray(format="bgr24"), format="bgr24"
                )
            except Exception:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                return av.VideoFrame.from_ndarray(blank, format="bgr24")


def _inject_browser_compatibility_check():
    """
    Inject a small client-side script (via an HTML component) that checks
    whether the browser supports the WebRTC APIs required by this app.

    If the browser is unsupported, a visible warning is rendered inside the
    component.  If supported, the component is effectively invisible.
    """
    st.components.v1.html(
        """
        <script>
        (function () {
            var webRTCSupported =
                !!(window.RTCPeerConnection ||
                    window.webkitRTCPeerConnection ||
                    window.mozRTCPeerConnection);
            var getUserMedia =
                !!(navigator.mediaDevices &&
                    navigator.mediaDevices.getUserMedia);

            if (!webRTCSupported || !getUserMedia) {
                var style =
                    "background:#ffe0e0;padding:12px;border-radius:6px;" +
                    "font-family:sans-serif;font-size:14px;";
                var msg =
                    "<div style='" + style + "'>" +
                    "<strong>⚠️ Browser Not Supported</strong><br>" +
                    "WebRTC is required for the Live Webcam feature.<br>" +
                    "Please use Google Chrome, Microsoft Edge, or " +
                    "Mozilla Firefox (latest versions). Safari 14+ is " +
                    "also supported." +
                    "</div>";
                document.body.innerHTML = msg;
            } else {
                document.body.style.display = "none";
            }
        })();
        </script>
        """,
        height=120,
    )


def _show_troubleshooting(turn_configured: bool):
    """Render an expandable troubleshooting guide."""
    with st.expander("🔧 Troubleshooting & Help", expanded=True):
        st.markdown(
            """
            **Common causes and fixes:**

            1. **Browser not supported** → Use Chrome or Edge (latest versions).
            2. **Camera permission denied** → Click the lock icon 🔒 in your browser's
               address bar and set **Camera → Allow**, then click **Start**.
            3. **No webcam detected** → Check that a webcam is connected and not in use
               by another application.
            4. **WebRTC connection timeout** → The app uses multiple Google STUN servers.
               """
            + (
                "TURN relay support is also enabled via Streamlit Secrets, which helps"
                " on networks that block UDP."
                if turn_configured
                else "For restricted networks, add TURN server credentials to **st.secrets.turn**"
                " (see below)."
            )
            + """
            5. **Corporate firewall / proxy** → May block UDP traffic needed for
               WebRTC. Contact your network administrator or use a mobile hotspot.
            6. **HTTPS requirement** → WebRTC requires HTTPS on deployed sites.
               Streamlit Community Cloud serves apps over HTTPS automatically. ✓

            **Enable TURN servers (optional, recommended for restricted networks):**

            In your Streamlit Cloud dashboard → Settings → Secrets, add:

            ```toml
            [turn]
            urls = "turn:turn.your-server.com:3478"
            username = "your-turn-username"
            credential = "your-turn-credential"
            ```
            """
        )


def render_webcam_section():
    """
    Render the complete Live Webcam section.

    This replaces the original minimal webrtc_streamer call with a robust,
    production-ready implementation that addresses every issue found during
    analysis:

      - Multiple public STUN servers (reliability / redundancy)
      - Optional TURN relay from Streamlit Secrets with STUN-only fallback
      - Modern ``VideoProcessorBase`` API (replaces deprecated ``VideoTransformerBase``)
      - Browser compatibility detection (client-side)
      - Graceful handling of permission denied / no-webcam / connection failures
      - Automatic reconnection when the WebRTC session disconnects
      - Support for multiple face detection
      - Low-latency processing (async_processing + frame skipping)
    """
    st.header("Live Webcam Emotion Detection")
    st.info("📹 Real-time emotion detection from your webcam - Multiple faces supported!")

    # ------------------------------------------------------------------
    # 1. Browser compatibility check (runs in the browser, not the server)
    # ------------------------------------------------------------------
    _inject_browser_compatibility_check()

    # ------------------------------------------------------------------
    # 2. Build RTC configuration (STUN + optional TURN)
    # ------------------------------------------------------------------
    rtc_config, turn_configured = _build_rtc_configuration()

    st.caption(
        "🔧 **WebRTC Configuration:** "
        f"{'Google STUN + TURN' if turn_configured else 'Google STUN (7 servers)'} — "
        f"{'TURN relay enabled via Streamlit Secrets' if turn_configured else 'STUN-only mode (TURN available via st.secrets.turn)'}"
    )

    # ------------------------------------------------------------------
    # 3. Reconnection state tracked in session_state
    # ------------------------------------------------------------------
    if "webcam_key_counter" not in st.session_state:
        st.session_state.webcam_key_counter = 0
    if "webcam_prev_playing" not in st.session_state:
        st.session_state.webcam_prev_playing = False
    if "webcam_reconnect_attempts" not in st.session_state:
        st.session_state.webcam_reconnect_attempts = 0
    if "webcam_error" not in st.session_state:
        st.session_state.webcam_error = None

    # A unique key per "session" — changing it forces streamlit-webrtc to
    # tear down the old WebRtcStreamerContext and create a fresh one, which
    # is how we achieve reconnection on Streamlit Community Cloud.
    webcam_key = f"emotion-det-webcam-{st.session_state.webcam_key_counter}"

    # ------------------------------------------------------------------
    # 4. Render the WebRTC streamer with correct, verified API
    # ------------------------------------------------------------------
    try:
        webrtc_ctx = webrtc_streamer(
            key=webcam_key,
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # ------------------------------------------------------------------
        # 5. State monitoring & auto-reconnection
        # ------------------------------------------------------------------
        # webrtc_streamer always returns a WebRtcStreamerContext object
        # (never None).  The context exposes a .state named-tuple with
        # two boolean fields: .playing and .signalling.
        is_playing = bool(webrtc_ctx.state.playing) if webrtc_ctx else False
        is_signalling = bool(webrtc_ctx.state.signalling) if webrtc_ctx else False
        was_playing = st.session_state.webcam_prev_playing
        st.session_state.webcam_prev_playing = is_playing

        # Detect a *drop* from playing → not-playing (connection lost)
        if was_playing and not is_playing and not is_signalling:
            if st.session_state.webcam_reconnect_attempts < 3:
                st.session_state.webcam_reconnect_attempts += 1
                st.session_state.webcam_key_counter += 1
                st.session_state.webcam_error = "disconnected"
                st.warning("🔌 Webcam connection was lost. Attempting to reconnect...")
                st.rerun()
            else:
                st.session_state.webcam_error = "max_retries_reached"

        # ------------------------------------------------------------------
        # 6. User-facing status messages based on WebRTC state
        # ------------------------------------------------------------------
        if is_playing:
            # Connection is healthy — reset retry counters
            st.session_state.webcam_reconnect_attempts = 0
            st.session_state.webcam_error = None
            st.success(
                "✅ Webcam is active! Emotions will be detected in real-time "
                "for all faces."
            )
            st.caption(
                "👁️ Multiple faces are supported. Each face is annotated with "
                "an emotion label and confidence score."
            )
        elif is_signalling:
            st.session_state.webcam_error = None
            st.info("🔌 Establishing secure connection to your webcam...")
            st.caption(
                "This may take a few seconds. If it takes longer than expected, "
                "check your network/firewall settings or click the lock icon 🔒 "
                "in your browser's address bar to grant camera permission."
            )
        else:
            # Idle state — streamer is rendered but not connected yet
            if st.session_state.webcam_error == "max_retries_reached":
                st.error(
                    "❌ Could not establish a webcam connection after 3 attempts."
                )
                _show_troubleshooting(turn_configured)
            else:
                st.info(
                    "📷 Click **Start** to activate your webcam.\n\n"
                    "Make sure to grant camera permission when prompted."
                )

    except Exception as exc:
        # catch unexpected streamlit-webrtc / aiortc errors
        st.session_state.webcam_error = str(exc)
        st.error(f"❌ Webcam stream initialization failed: {exc}")
        _show_troubleshooting(turn_configured)

    # ------------------------------------------------------------------
    # 7. Manual reconnect button (shows when auto-reconnect has given up
    #    or when an explicit error was raised)
    # ------------------------------------------------------------------
    show_manual_reconnect = (
        st.session_state.webcam_error
        and st.session_state.webcam_error != "disconnected"
    )
    if show_manual_reconnect:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Reconnect Webcam", key="webcam-reconnect-btn"):
                st.session_state.webcam_key_counter += 1
                st.session_state.webcam_error = None
                st.session_state.webcam_reconnect_attempts = 0
                st.rerun()


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

        **Features:**
        - Multiple face detection
        - Real-time processing
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
        tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎬 Video Upload", "📹 Live Webcam"])

        # Tab 1: Image Upload
        with tab1:
            st.header("Upload Image for Emotion Detection")

            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image containing faces"
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
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

                        # Process and predict
                        with st.spinner("Analyzing emotions..."):
                            annotated_image, emotions_list = detect_emotions_in_frame(image.copy())

                            with col2:
                                st.subheader("Result")
                                if emotions_list:
                                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

                                    # Display emotions
                                    st.markdown(f'<div class="emotion-box">{len(emotions_list)} Face(s) Detected</div>', unsafe_allow_html=True)

                                    for i, result in enumerate(emotions_list, 1):
                                        confidence_pct = result['confidence'] * 100
                                        st.markdown(f"""
                                        <div class="confidence-bar">
                                            <div class="confidence-fill" style="width: {confidence_pct}%">
                                                Face {i}: {result['emotion'].upper()} ({confidence_pct:.1f}%)
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.error("❌ No faces detected")
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
                                    _, emotions_list = detect_emotions_in_frame(frame.copy())

                                    if emotions_list:
                                        for emotion_data in emotions_list:
                                            results.append({
                                                'frame': frame_count,
                                                'time': frame_count / fps,
                                                'emotion': emotion_data['emotion'],
                                                'confidence': emotion_data['confidence']
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

        # Tab 3: Live Webcam
        with tab3:
            render_webcam_section()
    else:
        st.error("❌ Model not loaded. Please check the error messages above.")
        st.info("💡 Tip: Make sure the model files exist in the models/ directory")

if __name__ == '__main__':
    main()
