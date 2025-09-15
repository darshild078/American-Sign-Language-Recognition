import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import time
from typing import Optional, Tuple

class RobustCameraHandler:
    def __init__(self):
        """Ultra-reliable camera setup with auto-detection"""
        self.cap = None
        
        # Try all possible camera indices
        for camera_id in [0, 1, 2, 3]:
            try:
                test_cap = cv2.VideoCapture(camera_id)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        self.cap = test_cap
                        print(f"✅ Camera found at index {camera_id}")
                        break
                    test_cap.release()
            except:
                continue
        
        if not self.cap or not self.cap.isOpened():
            raise Exception("No working camera found!")
        
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # MediaPipe with bulletproof settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.frame_count = 0
        self.consecutive_detections = 0
        
    def get_frame(self):
        """Get frame with error handling"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror effect
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            return None
    
    def detect_and_extract_hand(self, frame):
        """Combined detection and extraction with visual feedback"""
        try:
            results = self.hands.process(frame)
            
            if not results.multi_hand_landmarks:
                self.consecutive_detections = 0
                return None, False, frame
            
            self.consecutive_detections += 1
            
            # Get hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on frame for visual feedback
            annotated_frame = frame.copy()
            self.mp_draw.draw_landmarks(
                annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Extract hand region
            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Generous padding
            pad = 60
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Extract hand region
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            if hand_region.size == 0:
                return None, False, annotated_frame
            
            # Only consider it detected after 3 consecutive frames
            is_stable = self.consecutive_detections >= 3
            
            return hand_region, is_stable, annotated_frame
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return None, False, frame
    
    def release(self):
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()

class AntibiasModelPredictor:
    def __init__(self, model):
        self.model = model
        self.input_size = (128, 128)
        
        # Anti-bias system
        self.recent_predictions = deque(maxlen=20)
        self.class_usage_count = Counter()
        
        # Problem classes that tend to dominate
        self.problematic_classes = {'E', 'nothing'}
        
    def preprocess_frame(self, frame):
        """Robust preprocessing"""
        try:
            if frame is None or frame.size == 0:
                return None
            
            # Resize and normalize
            frame_resized = cv2.resize(frame, self.input_size)
            
            # Quality check - ensure there's enough content
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
            if np.mean(gray) < 10:  # Too dark
                return None
            
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            return np.expand_dims(frame_normalized, axis=0)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict_with_antibias(self, processed_frame, class_names):
        """Prediction with built-in bias correction"""
        if processed_frame is None:
            return "nothing", 0.0
        
        try:
            # Get model predictions
            predictions = self.model.predict(processed_frame, verbose=0)
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            top_predictions = [(class_names[i], predictions[0][i]) for i in top_indices]
            
            print(f"Top predictions: {top_predictions}")
            
            # Anti-bias logic
            for class_name, confidence in top_predictions:
                
                # Skip if confidence too low
                if confidence < 0.3:
                    continue
                
                # Check if this class is overused recently
                recent_count = sum(1 for p in self.recent_predictions if p == class_name)
                
                # Penalize overused classes
                if class_name in self.problematic_classes and recent_count > 3:
                    print(f"❌ Penalizing overused class: {class_name}")
                    continue
                
                # Additional penalty for 'E' specifically
                if class_name == 'E' and recent_count > 1:
                    print(f"❌ Blocking excessive E predictions")
                    continue
                
                # Check overall usage
                if self.class_usage_count[class_name] > 15:  # Max 15 uses per session
                    print(f"❌ Class {class_name} used too much this session")
                    continue
                
                # Accept this prediction
                self.recent_predictions.append(class_name)
                self.class_usage_count[class_name] += 1
                
                print(f"✅ Accepted: {class_name} ({confidence:.2f})")
                return class_name, confidence
            
            # If all top predictions are blocked, return a safe default
            print("⚠️ All predictions blocked, returning 'nothing'")
            return "nothing", 0.0
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "nothing", 0.0
    
    def overlay_prediction(self, frame, prediction, confidence):
        """Add prediction overlay"""
        try:
            # Convert to BGR for OpenCV text
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                display_frame = frame.copy()
            
            # Color coding
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Main text
            text = f"{prediction.upper()} ({confidence:.0%})"
            cv2.putText(display_frame, text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Anti-bias indicator
            usage = self.class_usage_count.get(prediction, 0)
            usage_text = f"Used: {usage}/15"
            cv2.putText(display_frame, usage_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert back to RGB
            return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame

class UltraStableSmoother:
    def __init__(self):
        self.window_size = 8
        self.predictions = deque(maxlen=self.window_size)
        self.last_output = None
        self.last_output_time = 0
        self.min_interval = 2.5  # 2.5 seconds between outputs
        
    def add_prediction(self, prediction: str, confidence: float):
        """Ultra-stable smoothing with long intervals"""
        
        # Only accept high-confidence predictions
        if confidence < 0.6:
            return None
        
        current_time = time.time()
        
        # Enforce minimum time interval
        if (self.last_output and 
            current_time - self.last_output_time < self.min_interval):
            return None
        
        self.predictions.append(prediction)
        
        # Need strong majority
        if len(self.predictions) < 6:
            return None
        
        # Count occurrences
        counts = Counter(self.predictions)
        most_common = counts.most_common(1)[0]
        
        # Need super majority (5 out of 8)
        if most_common[1] >= 5:
            self.last_output = most_common[0]
            self.last_output_time = current_time
            print(f"🏆 FINAL OUTPUT: {most_common[0]}")
            return most_common[0]
        
        return None
    
    def reset(self):
        self.predictions.clear()
        self.last_output = None
        self.last_output_time = 0

# Aliases for compatibility
CameraHandler = RobustCameraHandler
ModelPredictor = AntibiasModelPredictor
PredictionSmoother = UltraStableSmoother
