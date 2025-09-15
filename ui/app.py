import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import mediapipe as mp
from typing import Optional, Tuple, List

class CameraHandler:
    """Optimized camera handler for real-time ASL detection"""
    
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # RELAXED MediaPipe settings for real-time video
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,           # CRITICAL: False for video streams
            max_num_hands=1,
            min_detection_confidence=0.6,     # LOWERED from 0.8
            min_tracking_confidence=0.5       # LOWERED from 0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Simplified validation - less strict for real-time
        self.frame_count = 0
        self.hand_detected_frames = 0
        self.min_hand_frames = 3  # REDUCED from 5
        
        # Motion detection for stability
        self.prev_frame = None
        self.motion_threshold = 30  # Lower = more sensitive to motion
        self.hand_region_buffer = deque(maxlen=10)
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Capture and return RGB frame with enhancement"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        
        # Flip frame horizontally for mirror effect (more natural for user)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhance frame for better detection
        frame_rgb = self.enhance_frame(frame_rgb)
        
        return frame_rgb
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame for better real-time detection"""
        
        # Reduce motion blur with slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel * 0.5)
        
        # Improve contrast and brightness
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2RGB)
        
        return frame
    
    def is_hand_stable(self, frame: np.ndarray) -> bool:
        """Check if hand movement is minimal (good for sign detection)"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False
        
        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        motion_level = np.mean(diff)
        
        self.prev_frame = gray
        
        # Return True if motion is low (stable hand)
        return motion_level < self.motion_threshold
    
    def detect_hand_simple(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
        """Simplified hand detection optimized for real-time"""
        
        results = self.hands.process(frame)
        
        if not results.multi_hand_landmarks:
            self.hand_detected_frames = max(0, self.hand_detected_frames - 1)
            return False, None, None
        
        # Get hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract hand region with generous padding
        h, w, _ = frame.shape
        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # GENEROUS padding for real-time (40% on each side)
        padding_x = int((x_max - x_min) * 0.4)
        padding_y = int((y_max - y_min) * 0.4)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        # RELAXED size validation
        hand_width = x_max - x_min
        hand_height = y_max - y_min
        hand_area = hand_width * hand_height
        
        # More permissive area check (2% to 70% of frame)
        min_area = (w * h) * 0.02
        max_area = (w * h) * 0.70
        
        if not (min_area <= hand_area <= max_area):
            return False, None, None
        
        # Extract hand region
        hand_region = frame[y_min:y_max, x_min:x_max]
        
        if hand_region.size == 0:
            return False, None, None
        
        self.hand_detected_frames += 1
        
        hand_info = {
            'bbox': (x_min, y_min, x_max, y_max),
            'confidence': 0.9,  # Assume good confidence if detected
            'landmarks': hand_landmarks
        }
        
        return True, hand_region, hand_info
    
    def detect_hand(self, frame: np.ndarray) -> bool:
        """Simple hand detection for basic validation"""
        hand_detected, _, _ = self.detect_hand_simple(frame)
        return hand_detected and self.hand_detected_frames >= self.min_hand_frames
    
    def get_hand_region_for_prediction(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get hand region optimized for real-time prediction"""
        
        # Only predict when hand is relatively stable
        if not self.is_hand_stable(frame):
            return None
        
        hand_detected, hand_region, hand_info = self.detect_hand_simple(frame)
        
        if not hand_detected or hand_region is None:
            return None
        
        return hand_region
    
    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()

class ModelPredictor:
    """Optimized model predictor for real-time use"""
    
    def __init__(self, model):
        self.model = model
        self.input_size = (128, 128)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimized preprocessing for real-time frames"""
        
        # Resize to model input size
        frame_resized = cv2.resize(frame, self.input_size)
        
        # Apply moderate denoising for real-time frames
        frame_resized = cv2.bilateralFilter(frame_resized, 5, 50, 50)
        
        # Normalize pixel values
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict_with_confidence(self, processed_frame: np.ndarray, class_names: List[str]) -> Tuple[Optional[str], float]:
        """Predict with relaxed confidence for real-time"""
        
        try:
            predictions = self.model.predict(processed_frame, verbose=0)
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            predicted_class = class_names[predicted_idx]
            
            # RELAXED confidence threshold for real-time (0.5 instead of 0.7)
            if confidence < 0.5:
                return None, confidence
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def predict(self, processed_frame: np.ndarray, class_names: List[str]) -> Tuple[str, float]:
        """Legacy predict method for compatibility"""
        prediction, confidence = self.predict_with_confidence(processed_frame, class_names)
        return prediction or "nothing", confidence
    
    def overlay_prediction(self, frame: np.ndarray, prediction: str, confidence: float) -> np.ndarray:
        """Enhanced overlay with real-time feedback"""
        frame_copy = frame.copy()
        
        # Color coding
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.5:
            color = (255, 255, 0)  # Yellow
        else:
            color = (255, 0, 0)  # Red
        
        # Add prediction text
        text = f"{prediction.upper()} ({confidence:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background
        cv2.rectangle(frame_copy, (10, 10), (text_width + 20, text_height + 30), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame_copy, text, (15, text_height + 15), font, font_scale, color, thickness)
        
        return frame_copy

class PredictionSmoother:
    """Relaxed prediction smoother for real-time use"""
    
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.5):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.predictions = deque(maxlen=window_size)
        self.last_stable_prediction = None
        self.last_prediction_time = 0
        self.min_time_between_predictions = 1.0  # REDUCED from 1.5
    
    def add_prediction(self, prediction: str, confidence: float) -> Optional[str]:
        """Relaxed smoothing for real-time detection"""
        
        if prediction is None or confidence < self.confidence_threshold:
            return None
        
        current_time = time.time()
        
        # Add to window
        self.predictions.append((prediction, confidence, current_time))
        
        # Need fewer predictions for real-time (3 instead of 5)
        if len(self.predictions) < 3:
            return None
        
        # Check for majority in recent predictions (last 3)
        recent_predictions = [p[0] for p in list(self.predictions)[-3:]]
        
        # Majority vote (2 out of 3 is enough)
        vote_counts = Counter(recent_predictions)
        most_common = vote_counts.most_common(1)[0]
        
        if most_common[1] >= 2:  # At least 2 out of 3 agree
            stable_prediction = most_common[0]
            
            # Check timing
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

class ASLSentenceBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Alphabet Sentence Builder")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f2f6')
        self.root.resizable(False, False)  # Prevent window resizing
        
        # Fixed dimensions for camera display
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        
        # Initialize variables
        self.camera_running = False
        self.camera_handler = None
        self.sentence = ""
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.hand_detected = False
        
        # Load model and initialize components
        self.load_model()
        self.smoother = PredictionSmoother(window_size=5, confidence_threshold=0.5)
        
        # Class names
        self.class_names = [
            'A','B','C','D','E','F','G','H','I','J','K','L','M',
            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'del','nothing','space'
        ]
        
        # Create GUI
        self.create_widgets()
        
        # Start camera update loop
        self.update_camera()
        
    def load_model(self):
        """Load the trained MobileNetV2 model"""
        try:
            model_path = '../checkpoints/best_mobilenet.h5'
            self.model = tf.keras.models.load_model(model_path)
            self.predictor = ModelPredictor(self.model)
            print("✅ Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        
        # Main container with fixed proportions
        main_frame = tk.Frame(self.root, bg='#f0f2f6')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left side - Camera feed (FIXED SIZE)
        left_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Camera title
        camera_title = tk.Label(left_frame, text="LIVE CAMERA FEED", 
                               font=('Arial', 16, 'bold'), bg='white', fg='#333')
        camera_title.pack(pady=10)
        
        # FIXED SIZE CAMERA CONTAINER - This prevents layout disruption
        self.camera_container = tk.Frame(left_frame, 
                                       width=self.CAMERA_WIDTH, 
                                       height=self.CAMERA_HEIGHT,
                                       bg='#808080', relief='solid', bd=2)
        self.camera_container.pack(pady=10, padx=20)
        
        # CRITICAL: Prevent the container from resizing based on its contents
        self.camera_container.pack_propagate(False)
        
        # Camera display label inside the fixed container
        self.camera_label = tk.Label(self.camera_container, bg='#808080', 
                                   text="Camera Off", font=('Arial', 20),
                                   fg='white')
        self.camera_label.pack(fill='both', expand=True)
        
        # Camera controls
        camera_controls = tk.Frame(left_frame, bg='white')
        camera_controls.pack(pady=10)
        
        self.start_btn = tk.Button(camera_controls, text="▶ Start Camera", 
                                  font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white',
                                  padx=20, pady=10, command=self.start_camera)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(camera_controls, text="⏹ Stop Camera", 
                                 font=('Arial', 12, 'bold'), bg='#f44336', fg='white',
                                 padx=20, pady=10, command=self.stop_camera)
        self.stop_btn.pack(side='left', padx=5)
        
        # Upload image button
        self.upload_btn = tk.Button(left_frame, text="UPLOAD IMAGE", 
                                   font=('Arial', 12, 'bold'), bg='#2196F3', fg='white',
                                   padx=30, pady=10, command=self.upload_image)
        self.upload_btn.pack(pady=10)
        
        # Right side - Predictions and Controls (FIXED WIDTH)
        right_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2, width=400)
        right_frame.pack(side='right', fill='y', padx=(10, 0))
        right_frame.pack_propagate(False)  # Prevent resizing
        
        # Prediction display
        pred_title = tk.Label(right_frame, text="Predicted ASL Sign:", 
                             font=('Arial', 14, 'bold'), bg='white', fg='#333')
        pred_title.pack(pady=10)
        
        # Current prediction box
        pred_frame = tk.Frame(right_frame, bg='#e8f4f8', relief='solid', bd=2)
        pred_frame.pack(pady=10, padx=20, fill='x')
        
        self.prediction_label = tk.Label(pred_frame, text="...", 
                                       font=('Arial', 24, 'bold'), bg='#e8f4f8', fg='#333',
                                       height=3)
        self.prediction_label.pack(pady=20)
        
        # Confidence display
        self.confidence_label = tk.Label(right_frame, text="Confidence: 0%", 
                                       font=('Arial', 12), bg='white', fg='#666')
        self.confidence_label.pack(pady=5)
        
        # Status display
        self.status_label = tk.Label(right_frame, text="Camera: Off", 
                                   font=('Arial', 12), bg='white', fg='#666')
        self.status_label.pack(pady=5)
        
        # Hand detection status
        self.hand_status_label = tk.Label(right_frame, text="Hand: Not Detected", 
                                        font=('Arial', 11), bg='white', fg='#999')
        self.hand_status_label.pack(pady=2)
        
        # Sentence builder
        sentence_title = tk.Label(right_frame, text="Built Sentence:", 
                                font=('Arial', 14, 'bold'), bg='white', fg='#333')
        sentence_title.pack(pady=(20, 10))
        
        # Sentence display
        sentence_frame = tk.Frame(right_frame, bg='#f9f9f9', relief='solid', bd=1)
        sentence_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        self.sentence_text = tk.Text(sentence_frame, font=('Arial', 12), 
                                   bg='#f9f9f9', fg='#333', wrap='word',
                                   height=8, padx=10, pady=10)
        self.sentence_text.pack(fill='both', expand=True)
        
        # Sentence controls
        sentence_controls = tk.Frame(right_frame, bg='white')
        sentence_controls.pack(pady=10, fill='x')
        
        clear_btn = tk.Button(sentence_controls, text="Clear", 
                             font=('Arial', 10, 'bold'), bg='#ff9800', fg='white',
                             padx=15, pady=5, command=self.clear_sentence)
        clear_btn.pack(side='left', padx=(20, 5))
        
        backspace_btn = tk.Button(sentence_controls, text="Backspace", 
                                font=('Arial', 10, 'bold'), bg='#607d8b', fg='white',
                                padx=15, pady=5, command=self.backspace)
        backspace_btn.pack(side='left', padx=5)
        
        # Add reset detection button
        reset_btn = tk.Button(sentence_controls, text="Reset", 
                            font=('Arial', 10, 'bold'), bg='#9c27b0', fg='white',
                            padx=15, pady=5, command=self.reset_detection)
        reset_btn.pack(side='right', padx=(5, 20))
        
    def start_camera(self):
        """Start camera capture"""
        try:
            if not self.camera_running:
                self.camera_handler = CameraHandler()
                self.camera_running = True
                self.hand_detected = False
                self.status_label.config(text="Camera: Starting...", fg='#ff9800')
                self.hand_status_label.config(text="Hand: Searching...", fg='#ff9800')
                self.smoother.reset()
                print("📷 Camera started")
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
            
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        self.hand_detected = False
        if self.camera_handler:
            self.camera_handler.release()
            self.camera_handler = None
        self.status_label.config(text="Camera: Off", fg='#666')
        self.hand_status_label.config(text="Hand: Not Detected", fg='#999')
        # Reset camera display
        self.camera_label.config(image='', text="Camera Off", bg='#808080')
        self.camera_label.image = None
        self.prediction_label.config(text="...")
        self.confidence_label.config(text="Confidence: 0%", fg='#666')
        print("📷 Camera stopped")
        
    def reset_detection(self):
        """Reset hand detection and prediction smoothing"""
        if self.camera_handler:
            self.camera_handler.hand_detected_frames = 0
            self.camera_handler.hand_region_buffer.clear()
        self.hand_detected = False
        self.smoother.reset()
        self.hand_status_label.config(text="Hand: Searching...", fg='#ff9800')
        self.prediction_label.config(text="...")
        self.confidence_label.config(text="Confidence: 0%", fg='#666')
        print("🔄 Detection reset")
        
    def upload_image(self):
        """Upload and predict from image file"""
        file_path = filedialog.askopenfilename(
            title="Select ASL Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = cv2.imread(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Check if hand is detected in uploaded image
                hand_detected, hand_region, hand_info = self.camera_handler.detect_hand_simple(image_rgb) if self.camera_handler else (False, None, None)
                
                if not hand_detected:
                    messagebox.showwarning("No Hand Detected", "No valid hand detected in the uploaded image. Please upload an image with a clear hand sign.")
                    return
                
                # Use segmented hand region for display and prediction
                if hand_region is not None:
                    # Display the hand region
                    display_image = cv2.resize(hand_region, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
                    photo = ImageTk.PhotoImage(Image.fromarray(display_image))
                    self.camera_label.config(image=photo, text='')
                    self.camera_label.image = photo
                    
                    # Predict
                    processed_frame = self.predictor.preprocess_frame(hand_region)
                    if processed_frame is not None:
                        prediction, confidence = self.predictor.predict_with_confidence(processed_frame, self.class_names)
                        
                        if prediction is not None:
                            # Update display
                            self.update_prediction_display(prediction, confidence)
                            self.add_to_sentence(prediction)
                        else:
                            messagebox.showinfo("Low Confidence", "The model couldn't make a confident prediction on this image.")
                else:
                    # Fallback to full image
                    display_image = cv2.resize(image_rgb, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
                    photo = ImageTk.PhotoImage(Image.fromarray(display_image))
                    self.camera_label.config(image=photo, text='')
                    self.camera_label.image = photo
                
            except Exception as e:
                messagebox.showerror("Upload Error", f"Failed to process image: {str(e)}")
    
    def update_camera(self):
        """Optimized camera update for real-time detection"""
        if self.camera_running and self.camera_handler:
            try:
                frame = self.camera_handler.get_frame()
                
                if frame is not None:
                    # Simple hand detection
                    if not self.hand_detected:
                        hand_detected = self.camera_handler.detect_hand(frame)
                        if hand_detected:
                            self.hand_detected = True
                            self.status_label.config(text="Camera: Hand Ready ✅", fg='#4CAF50')
                            self.hand_status_label.config(text="Hand: Detected ✅", fg='#4CAF50')
                        else:
                            self.status_label.config(text="Camera: Active", fg='#2196F3')
                            self.hand_status_label.config(text="Hand: Show your hand 👋", fg='#ff9800')
                    
                    # Predict more frequently but only when hand is stable
                    if self.hand_detected and self.camera_handler.frame_count % 5 == 0:  # Every 5 frames
                        
                        hand_region = self.camera_handler.get_hand_region_for_prediction(frame)
                        
                        if hand_region is not None:
                            processed_frame = self.predictor.preprocess_frame(hand_region)
                            prediction, confidence = self.predictor.predict_with_confidence(processed_frame, self.class_names)
                            
                            if prediction is not None:
                                # Apply smoothing
                                smoothed_prediction = self.smoother.add_prediction(prediction, confidence)
                                
                                if smoothed_prediction:
                                    self.update_prediction_display(smoothed_prediction, confidence)
                                    self.add_to_sentence(smoothed_prediction)
                                
                                # Show current prediction (even if not smoothed)
                                frame = self.predictor.overlay_prediction(frame, prediction, confidence)
                            
                            # Show stability indicator
                            if self.camera_handler.is_hand_stable(frame):
                                cv2.putText(frame, "STABLE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "MOVING", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        else:
                            # Hand region not suitable for prediction
                            if self.hand_detected:  # Reset if we lost the hand
                                self.hand_detected = False
                                self.hand_status_label.config(text="Hand: Lost, searching...", fg='#ff9800')
                    
                    # Display frame
                    frame_resized = cv2.resize(frame, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
                    photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
                    self.camera_label.config(image=photo, text='')
                    self.camera_label.image = photo
                    
            except Exception as e:
                print(f"Camera error: {e}")
        
        self.root.after(33, self.update_camera)
        
    def update_prediction_display(self, prediction, confidence):
        """Update the prediction display"""
        self.current_prediction = prediction
        self.prediction_confidence = confidence
        
        # Update prediction label
        if prediction == "...":
            display_text = "..."
        else:
            display_text = prediction if prediction != "nothing" else "NOTHING"
        
        self.prediction_label.config(text=display_text.upper())
        
        # Update confidence
        self.confidence_label.config(text=f"Confidence: {confidence:.0%}")
        
        # Color coding based on confidence
        if confidence > 0.7:
            color = '#4CAF50'  # Green - high confidence
        elif confidence > 0.5:
            color = '#ff9800'  # Orange - medium confidence
        else:
            color = '#f44336'  # Red - low confidence
            
        self.confidence_label.config(fg=color)
        
    def add_to_sentence(self, prediction):
        """Add prediction to sentence"""
        if prediction == "space":
            self.sentence += " "
        elif prediction == "del":
            if self.sentence:
                self.sentence = self.sentence[:-1]
        elif prediction != "nothing" and prediction != "...":
            self.sentence += prediction
            
        # Update sentence display
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.insert(1.0, self.sentence)
        
        # Auto-scroll to end
        self.sentence_text.see(tk.END)
        
    def clear_sentence(self):
        """Clear the built sentence"""
        self.sentence = ""
        self.sentence_text.delete(1.0, tk.END)
        
    def backspace(self):
        """Remove last character from sentence"""
        if self.sentence:
            self.sentence = self.sentence[:-1]
            self.sentence_text.delete(1.0, tk.END)
            self.sentence_text.insert(1.0, self.sentence)
            self.sentence_text.see(tk.END)
    
    def on_closing(self):
        """Handle application closing"""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ASLSentenceBuilder(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
