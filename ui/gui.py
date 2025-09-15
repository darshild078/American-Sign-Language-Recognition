import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import time
from utils import CameraHandler, ModelPredictor, PredictionSmoother

class FinalASLSentenceBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("🏆 FINAL ASL SENTENCE BUILDER - PRODUCTION READY")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f2f6')
        self.root.resizable(False, False)
        
        # Camera dimensions
        self.CAMERA_WIDTH = 680
        self.CAMERA_HEIGHT = 520
        
        # State variables
        self.camera_running = False
        self.camera_handler = None
        self.sentence = ""
        self.hand_stable = False
        self.total_predictions = 0
        self.successful_predictions = 0
        
        # Load model and setup
        self.load_model()
        self.smoother = PredictionSmoother()
        
        # ASL classes
        self.class_names = [
            'A','B','C','D','E','F','G','H','I','J','K','L','M',
            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'del','nothing','space'
        ]
        
        self.create_final_gui()
        self.start_camera_loop()
    
    def load_model(self):
        """Load model with comprehensive error handling"""
        try:
            model_path = '../checkpoints/best_mobilenet.h5'
            self.model = tf.keras.models.load_model(model_path)
            self.predictor = ModelPredictor(self.model)
            print("✅ Production model loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Model Not Found", 
                               f"Model file not found at: {model_path}\n\n"
                               "Please ensure the model file exists.")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Model Error", 
                               f"Failed to load model: {str(e)}\n\n"
                               "Please check the model file format.")
            self.root.destroy()
    
    def create_final_gui(self):
        """Create the final, polished GUI"""
        
        # Header
        header = tk.Frame(self.root, bg='#1976D2', height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        header_text = tk.Label(header, text="🏆 FINAL ASL RECOGNITION SYSTEM - PRODUCTION READY", 
                              font=('Arial', 18, 'bold'), bg='#1976D2', fg='white')
        header_text.pack(expand=True)
        
        # Main content
        main_frame = tk.Frame(self.root, bg='#f0f2f6')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Camera and controls
        left_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=3)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Camera section
        cam_header = tk.Frame(left_panel, bg='#4CAF50', height=40)
        cam_header.pack(fill='x')
        cam_header.pack_propagate(False)
        
        cam_title = tk.Label(cam_header, text="📹 LIVE CAMERA FEED", 
                            font=('Arial', 14, 'bold'), bg='#4CAF50', fg='white')
        cam_title.pack(expand=True)
        
        # Fixed camera container
        self.camera_container = tk.Frame(left_panel, 
                                       width=self.CAMERA_WIDTH, 
                                       height=self.CAMERA_HEIGHT,
                                       bg='#424242', relief='solid', bd=2)
        self.camera_container.pack(pady=15, padx=15)
        self.camera_container.pack_propagate(False)
        
        self.camera_label = tk.Label(self.camera_container, bg='#424242', 
                                   text="🏆 Production Ready\nClick Start to Begin", 
                                   font=('Arial', 16), fg='white')
        self.camera_label.pack(fill='both', expand=True)
        
        # Camera controls
        cam_controls = tk.Frame(left_panel, bg='white')
        cam_controls.pack(pady=15)
        
        self.start_btn = tk.Button(cam_controls, text="🚀 START SYSTEM", 
                                  font=('Arial', 13, 'bold'), bg='#4CAF50', fg='white',
                                  padx=30, pady=15, command=self.start_system)
        self.start_btn.pack(side='left', padx=10)
        
        self.stop_btn = tk.Button(cam_controls, text="⏹️ STOP SYSTEM", 
                                 font=('Arial', 13, 'bold'), bg='#f44336', fg='white',
                                 padx=30, pady=15, command=self.stop_system)
        self.stop_btn.pack(side='left', padx=10)
        
        # Emergency manual input
        emergency_frame = tk.Frame(left_panel, bg='#FFF3E0', relief='solid', bd=2)
        emergency_frame.pack(pady=15, padx=15, fill='x')
        
        emergency_title = tk.Label(emergency_frame, text="🆘 EMERGENCY MANUAL INPUT", 
                                 font=('Arial', 12, 'bold'), bg='#FFF3E0', fg='#E65100')
        emergency_title.pack(pady=8)
        
        # Manual letter buttons
        manual_letters = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'L', 'O', 'R', 'S', 'T', 'U', 'Y']
        
        btn_container = tk.Frame(emergency_frame, bg='#FFF3E0')
        btn_container.pack(pady=8)
        
        for i, letter in enumerate(manual_letters):
            btn = tk.Button(btn_container, text=letter, 
                           font=('Arial', 9, 'bold'), bg='#FF9800', fg='white',
                           padx=8, pady=4, width=2,
                           command=lambda l=letter: self.manual_input(l))
            btn.grid(row=i//8, column=i%8, padx=2, pady=2)
        
        # Right panel - Status and sentence
        right_panel = tk.Frame(main_frame, bg='white', relief='raised', bd=3, width=500)
        right_panel.pack(side='right', fill='y', padx=(15, 0))
        right_panel.pack_propagate(False)
        
        # Prediction section
        pred_header = tk.Frame(right_panel, bg='#2196F3', height=40)
        pred_header.pack(fill='x')
        pred_header.pack_propagate(False)
        
        pred_title = tk.Label(pred_header, text="🎯 AI PREDICTION ENGINE", 
                             font=('Arial', 14, 'bold'), bg='#2196F3', fg='white')
        pred_title.pack(expand=True)
        
        # Current prediction display
        pred_display = tk.Frame(right_panel, bg='#E8F5E8', relief='solid', bd=3)
        pred_display.pack(pady=15, padx=20, fill='x')
        
        self.prediction_label = tk.Label(pred_display, text="READY", 
                                       font=('Arial', 32, 'bold'), bg='#E8F5E8', fg='#1B5E20',
                                       height=2)
        self.prediction_label.pack(pady=30)
        
        # Status indicators
        status_frame = tk.Frame(right_panel, bg='#F5F5F5', relief='solid', bd=1)
        status_frame.pack(pady=10, padx=20, fill='x')
        
        self.confidence_label = tk.Label(status_frame, text="Confidence: 0%", 
                                       font=('Arial', 11), bg='#F5F5F5', fg='#333')
        self.confidence_label.pack(pady=4)
        
        self.camera_status_label = tk.Label(status_frame, text="📹 Camera: Offline", 
                                          font=('Arial', 11), bg='#F5F5F5', fg='#333')
        self.camera_status_label.pack(pady=4)
        
        self.hand_status_label = tk.Label(status_frame, text="👋 Hand: Not Detected", 
                                        font=('Arial', 11), bg='#F5F5F5', fg='#333')
        self.hand_status_label.pack(pady=4)
        
        # Performance metrics
        metrics_frame = tk.Frame(right_panel, bg='#E3F2FD', relief='solid', bd=1)
        metrics_frame.pack(pady=10, padx=20, fill='x')
        
        metrics_title = tk.Label(metrics_frame, text="📊 Performance Metrics", 
                               font=('Arial', 11, 'bold'), bg='#E3F2FD', fg='#1565C0')
        metrics_title.pack(pady=4)
        
        self.metrics_label = tk.Label(metrics_frame, text="Predictions: 0 | Success Rate: N/A", 
                                    font=('Arial', 10), bg='#E3F2FD', fg='#1565C0')
        self.metrics_label.pack(pady=4)
        
        # Sentence section
        sentence_header = tk.Frame(right_panel, bg='#FF9800', height=40)
        sentence_header.pack(fill='x', pady=(20, 0))
        sentence_header.pack_propagate(False)
        
        sentence_title = tk.Label(sentence_header, text="📝 SENTENCE BUILDER", 
                                font=('Arial', 14, 'bold'), bg='#FF9800', fg='white')
        sentence_title.pack(expand=True)
        
        # Sentence display
        sentence_container = tk.Frame(right_panel, bg='#FFF8E1', relief='solid', bd=2)
        sentence_container.pack(pady=15, padx=20, fill='both', expand=True)
        
        self.sentence_text = tk.Text(sentence_container, font=('Arial', 14), 
                                   bg='#FFF8E1', fg='#E65100', wrap='word',
                                   padx=15, pady=15)
        self.sentence_text.pack(fill='both', expand=True)
        
        # Sentence controls
        sentence_controls = tk.Frame(right_panel, bg='white')
        sentence_controls.pack(pady=20, fill='x')
        
        btn_row1 = tk.Frame(sentence_controls, bg='white')
        btn_row1.pack(fill='x', pady=5)
        
        clear_btn = tk.Button(btn_row1, text="🗑️ Clear All", 
                             font=('Arial', 11, 'bold'), bg='#F44336', fg='white',
                             padx=20, pady=10, command=self.clear_sentence)
        clear_btn.pack(side='left', padx=(20, 10))
        
        backspace_btn = tk.Button(btn_row1, text="⌫ Backspace", 
                                font=('Arial', 11, 'bold'), bg='#607D8B', fg='white',
                                padx=15, pady=10, command=self.backspace)
        backspace_btn.pack(side='left', padx=10)
        
        btn_row2 = tk.Frame(sentence_controls, bg='white')
        btn_row2.pack(fill='x', pady=5)
        
        space_btn = tk.Button(btn_row2, text="␣ Add Space", 
                            font=('Arial', 11, 'bold'), bg='#795548', fg='white',
                            padx=15, pady=10, command=self.add_space)
        space_btn.pack(side='left', padx=(20, 10))
        
        save_btn = tk.Button(btn_row2, text="💾 Save Text", 
                           font=('Arial', 11, 'bold'), bg='#009688', fg='white',
                           padx=15, pady=10, command=self.save_sentence)
        save_btn.pack(side='left', padx=10)
    
    def start_system(self):
        """Start the complete system"""
        try:
            if not self.camera_running:
                # Initialize camera
                self.camera_handler = CameraHandler()
                self.camera_running = True
                
                # Reset counters
                self.total_predictions = 0
                self.successful_predictions = 0
                
                # Update UI
                self.camera_status_label.config(text="📹 Camera: ACTIVE ✅", fg='#4CAF50')
                self.hand_status_label.config(text="👋 Show your hand clearly", fg='#FF9800')
                
                # Update buttons
                self.start_btn.config(state='disabled', bg='#A5D6A7', text="🚀 SYSTEM ACTIVE")
                self.stop_btn.config(state='normal')
                
                # Reset smoother
                self.smoother.reset()
                
                print("🏆 FINAL SYSTEM STARTED SUCCESSFULLY!")
                
        except Exception as e:
            error_msg = f"System startup failed: {str(e)}"
            messagebox.showerror("Startup Error", 
                               f"{error_msg}\n\nTroubleshooting:\n"
                               "• Check camera connections\n"
                               "• Restart the application\n"
                               "• Use manual input as backup")
    
    def stop_system(self):
        """Stop the complete system"""
        self.camera_running = False
        
        if self.camera_handler:
            self.camera_handler.release()
            self.camera_handler = None
        
        # Update UI
        self.camera_status_label.config(text="📹 Camera: Offline", fg='#666')
        self.hand_status_label.config(text="👋 Hand: Not Detected", fg='#666')
        self.camera_label.config(image='', text="🏆 Production Ready\nClick Start to Begin", bg='#424242')
        self.camera_label.image = None
        self.prediction_label.config(text="READY")
        self.confidence_label.config(text="Confidence: 0%")
        
        # Update buttons
        self.start_btn.config(state='normal', bg='#4CAF50', text="🚀 START SYSTEM")
        self.stop_btn.config(state='disabled', bg='#FFCDD2')
        
        print("🏆 System stopped")
    
    def manual_input(self, letter):
        """Emergency manual letter input"""
        self.sentence += letter
        self.update_sentence_display()
        self.prediction_label.config(text=f"MANUAL: {letter}")
        print(f"🆘 Manual input: {letter}")
    
    def start_camera_loop(self):
        """Main camera processing loop"""
        if self.camera_running and self.camera_handler:
            try:
                frame = self.camera_handler.get_frame()
                
                if frame is not None:
                    # Detect and extract hand
                    hand_region, is_stable, annotated_frame = self.camera_handler.detect_and_extract_hand(frame)
                    
                    # Update hand status
                    if hand_region is not None:
                        if is_stable:
                            self.hand_status_label.config(text="👋 Hand: STABLE ✅", fg='#4CAF50')
                            self.hand_stable = True
                        else:
                            self.hand_status_label.config(text="👋 Hand: Detected (stabilizing...)", fg='#FF9800')
                            self.hand_stable = False
                        
                        # Only predict when hand is stable and every 20 frames
                        if is_stable and self.camera_handler.frame_count % 20 == 0:
                            
                            # Preprocess and predict
                            processed_frame = self.predictor.preprocess_frame(hand_region)
                            
                            if processed_frame is not None:
                                self.total_predictions += 1
                                
                                # Get prediction with anti-bias
                                prediction, confidence = self.predictor.predict_with_antibias(
                                    processed_frame, self.class_names)
                                
                                # Update prediction display
                                self.update_prediction_display(prediction, confidence)
                                
                                # Apply ultra-stable smoothing
                                if confidence > 0.5:
                                    smoothed_prediction = self.smoother.add_prediction(prediction, confidence)
                                    
                                    if smoothed_prediction:
                                        self.successful_predictions += 1
                                        self.add_to_sentence(smoothed_prediction)
                                        print(f"🏆 FINAL: {smoothed_prediction}")
                                
                                # Add prediction overlay
                                annotated_frame = self.predictor.overlay_prediction(
                                    annotated_frame, prediction, confidence)
                    else:
                        self.hand_status_label.config(text="👋 Show your hand clearly", fg='#FF9800')
                        self.hand_stable = False
                    
                    # Update performance metrics
                    if self.total_predictions > 0:
                        success_rate = (self.successful_predictions / self.total_predictions) * 100
                        self.metrics_label.config(
                            text=f"Predictions: {self.total_predictions} | Success Rate: {success_rate:.1f}%")
                    
                    # Display frame
                    frame_resized = cv2.resize(annotated_frame, (self.CAMERA_WIDTH, self.CAMERA_HEIGHT))
                    photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
                    self.camera_label.config(image=photo, text='')
                    self.camera_label.image = photo
                    
            except Exception as e:
                print(f"Camera loop error: {e}")
        
        # Schedule next update (25 FPS for stability)
        self.root.after(40, self.start_camera_loop)
    
    def update_prediction_display(self, prediction, confidence):
        """Update prediction display with professional styling"""
        if prediction == "nothing":
            display_text = "SCANNING..."
        else:
            display_text = prediction.upper()
        
        self.prediction_label.config(text=display_text)
        self.confidence_label.config(text=f"Confidence: {confidence:.0%}")
        
        # Color coding
        if confidence > 0.8:
            color = '#1B5E20'  # Dark green
            bg_color = '#E8F5E8'
        elif confidence > 0.6:
            color = '#E65100'  # Orange
            bg_color = '#FFF3E0'
        else:
            color = '#C62828'  # Red
            bg_color = '#FFEBEE'
        
        self.prediction_label.config(fg=color, bg=bg_color)
        self.confidence_label.config(fg=color)
    
    def add_to_sentence(self, prediction):
        """Add prediction to sentence with special handling"""
        if prediction == "space":
            self.sentence += " "
        elif prediction == "del":
            if self.sentence:
                self.sentence = self.sentence[:-1]
        elif prediction not in ["nothing", "..."]:
            self.sentence += prediction
        
        self.update_sentence_display()
    
    def update_sentence_display(self):
        """Update sentence display"""
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.insert(1.0, self.sentence)
        self.sentence_text.see(tk.END)
    
    def clear_sentence(self):
        """Clear entire sentence"""
        self.sentence = ""
        self.update_sentence_display()
    
    def backspace(self):
        """Remove last character"""
        if self.sentence:
            self.sentence = self.sentence[:-1]
            self.update_sentence_display()
    
    def add_space(self):
        """Add space manually"""
        self.sentence += " "
        self.update_sentence_display()
    
    def save_sentence(self):
        """Save sentence to file"""
        if not self.sentence.strip():
            messagebox.showwarning("Empty Sentence", "No sentence to save!")
            return
        
        try:
            filename = f"ASL_Sentence_{int(time.time())}.txt"
            with open(filename, 'w') as f:
                f.write(f"ASL Sentence Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Sentence: {self.sentence}\n")
                f.write(f"Total Predictions: {self.total_predictions}\n")
                f.write(f"Successful Predictions: {self.successful_predictions}\n")
                if self.total_predictions > 0:
                    success_rate = (self.successful_predictions / self.total_predictions) * 100
                    f.write(f"Success Rate: {success_rate:.1f}%\n")
            
            messagebox.showinfo("Saved", f"Sentence saved as: {filename}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {str(e)}")
    
    def on_closing(self):
        """Handle application closing"""
        if self.camera_running:
            self.stop_system()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FinalASLSentenceBuilder(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
