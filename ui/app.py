import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import time
import threading
from utils import CameraHandler, ModelPredictor, PredictionSmoother

# Configure Streamlit page
st.set_page_config(
    page_title="ASL Sentence Builder",
    page_icon="🤟",
    layout="wide"
)

# Load MobileNetV2 model once
@st.cache_resource
def load_model():
    model_path = '../checkpoints/best_mobilenet.h5'
    return tf.keras.models.load_model(model_path)

# Class names for ASL alphabet
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

st.title("🤟 ASL Alphabet Sentence Builder")
st.markdown("Real-time ASL alphabet recognition with continuous capture")

# Initialize ALL session state variables FIRST (before any usage)
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'hand_detected' not in st.session_state:
    st.session_state.hand_detected = False
if 'camera_handler' not in st.session_state:
    st.session_state.camera_handler = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ""
if 'prediction_confidence' not in st.session_state:
    st.session_state.prediction_confidence = 0.0

# Initialize components AFTER session state is ready
model = load_model()
predictor = ModelPredictor(model)
smoother = PredictionSmoother(window_size=5, confidence_threshold=0.7)

# Layout: Camera on left, controls on right
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Placeholders for dynamic content
    camera_placeholder = st.empty()
    status_placeholder = st.empty()
    prediction_placeholder = st.empty()

with col2:
    st.subheader("📝 Sentence Builder")
    
    # Display current sentence
    sentence_display = st.text_area(
        "Current Sentence:",
        value=st.session_state.sentence,
        height=150,
        key="sentence_display"
    )
    
    # Control buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("▶️ Start", type="primary")
    with col_btn2:
        stop_btn = st.button("⏹️ Stop")
    
    col_btn3, col_btn4 = st.columns(2)
    with col_btn3:
        clear_btn = st.button("🗑️ Clear")
    with col_btn4:
        backspace_btn = st.button("⌫ Backspace")
    
    # Statistics
    st.subheader("📊 Stats")
    if st.session_state.last_prediction:
        st.metric("Last Prediction", st.session_state.last_prediction)
        st.metric("Confidence", f"{st.session_state.prediction_confidence:.2f}")

# Camera thread function
def run_camera():
    """Background thread function for camera capture"""
    while st.session_state.camera_running:
        try:
            # Get frame from camera
            frame = st.session_state.camera_handler.get_frame()
            if frame is None:
                continue
            
            # Check for hand detection on startup
            if not st.session_state.hand_detected:
                hand_detected = st.session_state.camera_handler.detect_hand(frame)
                if hand_detected:
                    st.session_state.hand_detected = True
                    status_placeholder.success("✅ Hand detected! Starting recognition...")
                else:
                    status_placeholder.warning("👋 Please show your hand to start recognition")
                    camera_placeholder.image(frame, channels="RGB", use_container_width=True)
                    time.sleep(0.1)
                    continue
            
            # Predict every 5th frame (~6 FPS for predictions at 30 FPS capture)
            if st.session_state.camera_handler.frame_count % 5 == 0:
                # Preprocess and predict
                processed_frame = predictor.preprocess_frame(frame)
                prediction, confidence = predictor.predict(processed_frame, class_names)
                
                # Apply smoothing
                smoothed_prediction = smoother.add_prediction(prediction, confidence)
                
                if smoothed_prediction:
                    # Update sentence based on prediction
                    if smoothed_prediction == "space":
                        st.session_state.sentence += " "
                    elif smoothed_prediction == "del":
                        if st.session_state.sentence:
                            st.session_state.sentence = st.session_state.sentence[:-1]
                    elif smoothed_prediction != "nothing":
                        st.session_state.sentence += smoothed_prediction
                    
                    # Update display states
                    st.session_state.last_prediction = smoothed_prediction
                    st.session_state.prediction_confidence = confidence
                    
                    # Show prediction on frame
                    frame = predictor.overlay_prediction(frame, smoothed_prediction, confidence)
            
            # Update camera feed
            camera_placeholder.image(frame, channels="RGB", use_container_width=True)
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.session_state.camera_running = False
            break

# Handle button clicks
if start_btn and not st.session_state.camera_running:
    try:
        st.session_state.camera_running = True
        st.session_state.camera_handler = CameraHandler()
        
        # Start camera in background thread
        camera_thread = threading.Thread(target=run_camera, daemon=True)
        camera_thread.start()
        
        status_placeholder.info("📷 Camera starting...")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to start camera: {str(e)}")
        st.session_state.camera_running = False

if stop_btn and st.session_state.camera_running:
    st.session_state.camera_running = False
    st.session_state.hand_detected = False
    if st.session_state.camera_handler:
        st.session_state.camera_handler.release()
        st.session_state.camera_handler = None
    status_placeholder.info("📷 Camera stopped")
    smoother.reset()  # Reset prediction smoother
    st.rerun()

if clear_btn:
    st.session_state.sentence = ""
    st.rerun()

if backspace_btn and st.session_state.sentence:
    st.session_state.sentence = st.session_state.sentence[:-1]
    st.rerun()

# Show current status when camera is not running
if not st.session_state.camera_running:
    camera_placeholder.info("📷 Click 'Start' to begin ASL recognition")
    status_placeholder.info("Ready to start camera")