import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
from collections import deque, Counter
from typing import Optional

class PredictionSmoother:
    """
    5-frame smoothing, 3/5 majority, 5s cooldown, 1s warmup
    """
    def __init__(self, window_size: int = 5, majority_threshold: int = 3, cooldown_seconds: float = 5.0):
        self.window_size = window_size
        self.majority_threshold = majority_threshold
        self.cooldown = cooldown_seconds
        self.predictions = deque(maxlen=self.window_size)
        self.last_prediction_time = 0.0
        # NEW: warmup deadline; if now < _warmup_until, ignore predictions
        self._warmup_until = 0.0

    # NEW: call when a hand appears for the first time
    def start_warmup(self, seconds: float = 5.0):
        self._warmup_until = time.time() + seconds

    # NEW: force cooldown to start right now
    def force_cooldown(self):
        self.last_prediction_time = time.time()

    def add_prediction(self, prediction: Optional[str]) -> Optional[str]:
        now = time.time()

        # Block predictions during warmup window
        if now < self._warmup_until:
            return None

        # Standard cooldown
        if now - self.last_prediction_time < self.cooldown:
            return None

        # Push into window
        self.predictions.append('nothing' if prediction is None else prediction)

        # Need a full window
        if len(self.predictions) < self.window_size:
            return None

        # Majority vote
        counts = Counter(self.predictions)
        winner, votes = max(counts.items(), key=lambda kv: kv[1])

        # Clear the window and accept only if criteria met
        self.predictions.clear()

        if votes >= self.majority_threshold and winner != 'nothing':
            self.last_prediction_time = now
            return winner

        return None

    def get_cooldown_status(self) -> tuple[bool, float]:
        elapsed = time.time() - self.last_prediction_time
        in_cooldown = elapsed < self.cooldown
        remain = self.cooldown - elapsed if in_cooldown else 0.0
        return in_cooldown, remain

    def reset(self):
        self.predictions.clear()
        self.last_prediction_time = 0.0
        self._warmup_until = 0.0

class ASLSentenceBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("SignScribe - ASL Alphabet Recognition")
        self.root.geometry("1280x720")
        
        # --- UI Theme and Styling ---
        self.style = ttk.Style(self.root)
        self.style.theme_use('arc')

        # --- Colors ---
        self.COLOR_BG = '#F0F2F5'
        self.COLOR_FRAME = '#FFFFFF'
        self.COLOR_TEXT = '#2E353B'
        self.COLOR_PRIMARY = '#007ACC'
        self.COLOR_SUCCESS = '#28A745'
        self.COLOR_WARN = '#FFC107'
        self.COLOR_DANGER = '#DC3545'
        self.COLOR_DISABLED = '#B0B0B0'

        # --- Fonts ---
        self.FONT_BOLD = ("Segoe UI", 12, "bold")
        self.FONT_NORMAL = ("Segoe UI", 11)
        self.FONT_LARGE = ("Segoe UI", 48, "bold")
        self.FONT_TITLE = ("Segoe UI", 16, "bold")

        self.root.configure(bg=self.COLOR_BG)
        
        self.CAMERA_WIDTH, self.CAMERA_HEIGHT = 640, 480
        self.camera_running = False
        self.camera_handler = None
        self.sentence = ""
        
        self.load_model()
        self.smoother = PredictionSmoother()
        self.create_widgets()
        self.update_camera()

    def load_model(self):
        try:
            model_path = '../checkpoints/asl_landmark_model.pkl'
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✅ Landmark model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            self.root.destroy()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, style='TFrame', padding=20)
        main_frame.pack(fill='both', expand=True)

        # --- Left Panel: Camera ---
        left_panel = ttk.Frame(main_frame, style='Card.TFrame', padding=20)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))

        ttk.Label(left_panel, text="LIVE FEED", font=self.FONT_TITLE, foreground=self.COLOR_TEXT).pack(pady=(0, 15))

        self.camera_label = tk.Label(left_panel, bg='#000000', text="Camera Off", font=self.FONT_BOLD, fg='white')
        self.camera_label.pack(fill='both', expand=True, pady=5)
        
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(pady=(15, 0), fill='x')

        self.start_btn = ttk.Button(controls_frame, text="▶ Start Camera", command=self.start_camera, style='Success.TButton')
        self.start_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        self.stop_btn = ttk.Button(controls_frame, text="⏹ Stop Camera", command=self.stop_camera, style='Danger.TButton', state='disabled')
        self.stop_btn.pack(side='left', expand=True, fill='x', padx=5)

        # --- Right Panel: Controls & Output ---
        right_panel = ttk.Frame(main_frame, style='Card.TFrame', padding=20)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))

        ttk.Label(right_panel, text="RECOGNITION", font=self.FONT_TITLE, foreground=self.COLOR_TEXT).pack(pady=(0, 10))

        self.prediction_label = ttk.Label(right_panel, text="...", font=self.FONT_LARGE, foreground=self.COLOR_PRIMARY, anchor='center')
        self.prediction_label.pack(pady=20, fill='x')
        
        self.status_label = ttk.Label(right_panel, text="Status: Idle", font=self.FONT_NORMAL, foreground=self.COLOR_DISABLED, anchor='center')
        self.status_label.pack(pady=(0, 20), fill='x')
        
        ttk.Separator(right_panel, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(right_panel, text="SENTENCE", font=self.FONT_TITLE, foreground=self.COLOR_TEXT).pack(pady=10)
        
        sentence_frame = ttk.Frame(right_panel, style='Card.TFrame', padding=5)
        sentence_frame.pack(pady=5, fill='both', expand=True)

        self.sentence_text = tk.Text(sentence_frame, font=("Segoe UI", 14), wrap='word', bd=0, bg=self.COLOR_FRAME, fg=self.COLOR_TEXT, relief='flat', padx=10, pady=10)
        self.sentence_text.pack(fill='both', expand=True)
        
        sentence_controls = ttk.Frame(right_panel)
        sentence_controls.pack(pady=(15, 0), fill='x')
        
        ttk.Button(sentence_controls, text="Clear", command=self.clear_sentence, style='Warning.TButton').pack(side='left', expand=True, fill='x', padx=(0,5))
        ttk.Button(sentence_controls, text="Backspace", command=self.backspace, style='Secondary.TButton').pack(side='left', expand=True, fill='x', padx=5)

    def start_camera(self):
        try:
            self.camera_handler = cv2.VideoCapture(0)
            if not self.camera_handler.isOpened(): raise IOError("Cannot open webcam")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            self.camera_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.smoother.reset()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")

    def stop_camera(self):
        self.camera_running = False
        if self.camera_handler: self.camera_handler.release()
        if hasattr(self, 'hands'): self.hands.close()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.camera_label.config(image='', text="Camera Off", bg='black')
        self.camera_label.image = None
        self.prediction_label.config(text="...")
        self.status_label.config(text="Status: Idle", foreground=self.COLOR_DISABLED)

    def normalize_landmarks(self, landmarks):
        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        base = landmarks_np[0].copy()
        relative = landmarks_np - base
        max_val = np.max(np.abs(relative))
        return (relative / max_val).flatten().reshape(1, -1) if max_val > 0 else relative.flatten().reshape(1, -1)

    def update_camera(self):
        if self.camera_running and self.camera_handler:
            ret, frame = self.camera_handler.read()
            if ret:
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)
                
                raw_prediction = None
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    feature_vector = self.normalize_landmarks(hand_landmarks.landmark)
                    raw_prediction = self.model.predict(feature_vector)[0]
                    self.prediction_label.config(text=raw_prediction.upper())
                else:
                    self.prediction_label.config(text="...")

                is_cooldown, remaining = self.smoother.get_cooldown_status()
                if is_cooldown:
                    self.status_label.config(text=f"⌛ Cooldown ({remaining:.1f}s)", foreground=self.COLOR_DANGER)
                elif not results.multi_hand_landmarks:
                    self.status_label.config(text="🔍 Searching for hand...", foreground=self.COLOR_WARN)
                else:
                    self.status_label.config(text="✅ Ready to Detect", foreground=self.COLOR_SUCCESS)

                stable_prediction = self.smoother.add_prediction(raw_prediction)
                if stable_prediction: self.add_to_sentence(stable_prediction)

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = img_tk
                self.camera_label.config(image=img_tk)

        self.root.after(30, self.update_camera)

    def add_to_sentence(self, p):
        if p == "space": self.sentence += " "
        elif p == "del": self.backspace()
        else: self.sentence += p
        self.update_sentence_display()

    def update_sentence_display(self):
        self.sentence_text.delete(1.0, 'end')
        self.sentence_text.insert(1.0, self.sentence)
        self.sentence_text.see('end')

    def clear_sentence(self):
        self.sentence = ""
        self.update_sentence_display()

    def backspace(self):
        if self.sentence:
            self.sentence = self.sentence[:-1]
            self.update_sentence_display()
            
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

def main():
    root = ThemedTk(theme="arc")
    ASLSentenceBuilder(root)
    root.mainloop()

if __name__ == "__main__":
    main()
