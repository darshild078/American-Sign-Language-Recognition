import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from typing import Optional, Tuple, List

class CameraHandler:
    """Handles camera operations with MediaPipe hand detection"""
    
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.frame_count = 0
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture and return RGB frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def detect_hand(self, frame: np.ndarray) -> bool:
        """Detect if a hand is present in the frame"""
        results = self.hands.process(frame)
        return results.multi_hand_landmarks is not None
    
    def get_hand_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract hand region from frame (optional optimization)"""
        results = self.hands.process(frame)
        if not results.multi_hand_landmarks:
            return None
        
        # Get hand landmarks and create bounding box
        landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        
        x_coords = [landmark.x * w for landmark in landmarks.landmark]
        y_coords = [landmark.y * h for landmark in landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return frame[y_min:y_max, x_min:x_max]
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()

class ModelPredictor:
    """Handles model predictions and preprocessing"""
    
    def __init__(self, model):
        self.model = model
        self.input_size = (128, 128)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for MobileNetV2 input"""
        # Resize to model input size
        frame_resized = cv2.resize(frame, self.input_size)
        
        # Normalize pixel values
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict(self, processed_frame: np.ndarray, class_names: List[str]) -> Tuple[str, float]:
        """Make prediction and return class name with confidence"""
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = class_names[predicted_idx]
        
        return predicted_class, confidence
    
    def overlay_prediction(self, frame: np.ndarray, prediction: str, confidence: float) -> np.ndarray:
        """Overlay prediction text on frame"""
        frame_copy = frame.copy()
        
        # Add text overlay
        text = f"{prediction} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0)
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame_copy, (10, 10), (text_width + 20, text_height + 20), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame_copy, text, (15, text_height + 15), font, font_scale, color, thickness)
        
        return frame_copy

class PredictionSmoother:
    """Smooths predictions using sliding window and confidence thresholding"""
    
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.7):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.predictions = deque(maxlen=window_size)
        self.last_stable_prediction = None
        self.last_prediction_time = 0
        self.min_time_between_predictions = 1.0  # Minimum 1 second between accepted predictions
    
    def add_prediction(self, prediction: str, confidence: float) -> Optional[str]:
        """Add prediction and return smoothed result if stable"""
        current_time = time.time()
        
        # Only consider high-confidence predictions
        if confidence < self.confidence_threshold:
            return None
        
        # Add to sliding window
        self.predictions.append((prediction, confidence, current_time))
        
        # Need minimum number of predictions
        if len(self.predictions) < self.window_size:
            return None
        
        # Check if majority of recent predictions agree
        recent_predictions = [p[0] for p in list(self.predictions)[-3:]]  # Last 3 predictions
        if len(set(recent_predictions)) == 1:  # All agree
            stable_prediction = recent_predictions[0]
            
            # Avoid repeated predictions too quickly
            if (stable_prediction != self.last_stable_prediction or 
                current_time - self.last_prediction_time > self.min_time_between_predictions):
                
                self.last_stable_prediction = stable_prediction
                self.last_prediction_time = current_time
                return stable_prediction
        
        return None
    
    def reset(self):
        """Reset the smoother"""
        self.predictions.clear()
        self.last_stable_prediction = None
        self.last_prediction_time = 0