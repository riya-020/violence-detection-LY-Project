"""
Weapon Detection Application
Supports: Image Upload, Video Upload, Live Webcam Detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import os

class WeaponDetectionApp:
    def __init__(self, model_path):
        """
        Initialize the Weapon Detection Application
        
        Args:
            model_path: Path to your trained best.pt model
        """
        print("Loading weapon detection model...")
        self.model = YOLO(model_path)
        print("✓ Model loaded successfully!")
        
        # Get class names from model
        self.class_names = self.model.names
        print(f"Detected classes: {self.class_names}")
        
        # Detection settings
        self.confidence_threshold = 0.25
        self.detection_active = False
        self.video_capture = None
        
        # Alert settings
        self.alert_sound_enabled = True
        self.last_alert_time = 0
        self.alert_cooldown = 2  # seconds between alerts
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Create the main GUI window"""
        self.root = tk.Tk()
        self.root.title("🔫 Weapon Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="🛡️ WEAPON DETECTION SYSTEM",
            font=("Arial", 24, "bold"),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        title_label.pack(pady=20)
        
        # Button Frame
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=10)
        
        # Buttons
        btn_style = {
            'font': ('Arial', 12, 'bold'),
            'width': 15,
            'height': 2,
            'bg': '#4CAF50',
            'fg': 'white',
            'relief': 'raised',
            'bd': 3
        }
        
        self.upload_image_btn = tk.Button(
            button_frame,
            text="📷 Upload Image",
            command=self.upload_image,
            **btn_style
        )
        self.upload_image_btn.grid(row=0, column=0, padx=10)
        
        self.upload_video_btn = tk.Button(
            button_frame,
            text="🎥 Upload Video",
            command=self.upload_video,
            **btn_style
        )
        self.upload_video_btn.grid(row=0, column=1, padx=10)
        
        self.webcam_btn = tk.Button(
            button_frame,
            text="📹 Start Webcam",
            command=self.toggle_webcam,
            bg='#2196F3',
            **{k:v for k,v in btn_style.items() if k != 'bg'}
        )
        self.webcam_btn.grid(row=0, column=2, padx=10)
        
        # Confidence slider
        slider_frame = tk.Frame(self.root, bg='#2b2b2b')
        slider_frame.pack(pady=10)
        
        tk.Label(
            slider_frame,
            text="Confidence Threshold:",
            font=("Arial", 12),
            bg='#2b2b2b',
            fg='white'
        ).pack(side=tk.LEFT, padx=10)
        
        self.confidence_slider = tk.Scale(
            slider_frame,
            from_=0.1,
            to=0.9,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=300,
            bg='#4CAF50',
            fg='white',
            font=("Arial", 10),
            command=self.update_confidence
        )
        self.confidence_slider.set(self.confidence_threshold)
        self.confidence_slider.pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(
            slider_frame,
            text=f"{self.confidence_threshold:.2f}",
            font=("Arial", 12, "bold"),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        self.confidence_label.pack(side=tk.LEFT, padx=10)
        
        # Display Frame
        display_frame = tk.Frame(self.root, bg='#1a1a1a', relief='sunken', bd=3)
        display_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            display_frame,
            bg='#1a1a1a',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status Frame
        status_frame = tk.Frame(self.root, bg='#2b2b2b')
        status_frame.pack(pady=10, fill=tk.X)
        
        self.status_label = tk.Label(
            status_frame,
            text="📊 Status: Ready",
            font=("Arial", 12),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        self.status_label.pack()
        
        self.detection_label = tk.Label(
            status_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='#2b2b2b',
            fg='#ff0000'
        )
        self.detection_label.pack(pady=5)
        
        # Protocol for window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def update_confidence(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = float(value)
        self.confidence_label.config(text=f"{self.confidence_threshold:.2f}")
        
    def upload_image(self):
        """Handle image upload and detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.status_label.config(text="📊 Status: Processing image...")
            self.root.update()
            
            # Read and process image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Failed to load image!")
                return
                
            # Run detection
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Draw detections
            annotated_image = results[0].plot()
            
            # Check for weapons and show alert
            detections = self.process_detections(results[0])
            
            # Display image
            self.display_image(annotated_image)
            
            # Update status
            if detections:
                self.show_weapon_alert(detections)
                self.status_label.config(
                    text=f"⚠️ Status: {len(detections)} weapon(s) detected!",
                    fg='#ff0000'
                )
            else:
                self.status_label.config(
                    text="✓ Status: No weapons detected",
                    fg='#00ff00'
                )
                self.detection_label.config(text="")
                
    def upload_video(self):
        """Handle video upload and detection"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_video(file_path)
            
    def toggle_webcam(self):
        """Toggle webcam on/off"""
        if not self.detection_active:
            self.start_webcam()
        else:
            self.stop_detection()
            
    def start_webcam(self):
        """Start webcam detection"""
        self.video_capture = cv2.VideoCapture(0)
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Failed to open webcam!")
            return
            
        self.detection_active = True
        self.webcam_btn.config(
            text="⏹️ Stop Webcam",
            bg='#f44336'
        )
        self.status_label.config(text="📹 Status: Webcam active")
        
        # Start detection thread
        thread = threading.Thread(target=self.webcam_detection_loop, daemon=True)
        thread.start()
        
    def process_video(self, video_path):
        """Process video file"""
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Failed to open video!")
            return
            
        self.detection_active = True
        self.webcam_btn.config(text="⏹️ Stop Video", bg='#f44336')
        self.status_label.config(text="🎥 Status: Processing video...")
        
        # Start detection thread
        thread = threading.Thread(target=self.video_detection_loop, daemon=True)
        thread.start()
        
    def webcam_detection_loop(self):
        """Main loop for webcam detection"""
        while self.detection_active:
            ret, frame = self.video_capture.read()
            
            if not ret:
                break
                
            # Run detection
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Check for weapons
            detections = self.process_detections(results[0])
            
            # Display frame
            self.display_image(annotated_frame)
            
            # Update detection status
            if detections:
                self.show_weapon_alert(detections)
            else:
                self.detection_label.config(text="")
                
            # Small delay
            time.sleep(0.03)
            
        self.stop_detection()
        
    def video_detection_loop(self):
        """Main loop for video file detection"""
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03
        
        while self.detection_active:
            ret, frame = self.video_capture.read()
            
            if not ret:
                messagebox.showinfo("Complete", "Video processing complete!")
                break
                
            # Run detection
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Check for weapons
            detections = self.process_detections(results[0])
            
            # Display frame
            self.display_image(annotated_frame)
            
            # Update detection status
            if detections:
                self.show_weapon_alert(detections)
            else:
                self.detection_label.config(text="")
                
            # Maintain video speed
            time.sleep(delay)
            
        self.stop_detection()
        
    def process_detections(self, result):
        """Process detection results and return list of detected weapons"""
        detections = []
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                weapon_name = self.class_names[cls]
                
                detections.append({
                    'weapon': weapon_name,
                    'confidence': conf,
                    'box': box.xyxy[0].tolist()
                })
                
        return detections
        
    def show_weapon_alert(self, detections):
        """Show alert when weapon is detected"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
            
        # Create alert message
        alert_text = "⚠️ WEAPON DETECTED!\n"
        for det in detections:
            alert_text += f"• {det['weapon']}: {det['confidence']:.1%}\n"
            
        # Update label
        self.detection_label.config(text=alert_text)
        
        # Play alert sound (optional - requires pygame or winsound)
        try:
            import winsound
            winsound.Beep(1000, 500)  # Frequency, Duration
        except:
            pass
        
        self.last_alert_time = current_time
        
        # Log detection
        self.log_detection(detections)
        
    def log_detection(self, detections):
        """Log detections to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_dir = "weapon_detections"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "detection_log.txt")
        
        with open(log_file, "a") as f:
            f.write(f"\n[{timestamp}]\n")
            for det in detections:
                f.write(f"  - {det['weapon']}: {det['confidence']:.2%}\n")
                
    def display_image(self, image):
        """Display image on canvas"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate scaling
            img_height, img_width = image_rgb.shape[:2]
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            image_resized = cv2.resize(image_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_resized)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=photo,
                anchor=tk.CENTER
            )
            self.canvas.image = photo  # Keep reference
            
    def stop_detection(self):
        """Stop video/webcam detection"""
        self.detection_active = False
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        self.webcam_btn.config(
            text="📹 Start Webcam",
            bg='#2196F3'
        )
        self.status_label.config(text="📊 Status: Ready")
        self.detection_label.config(text="")
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()
        
    def run(self):
        """Start the application"""
        print("\n" + "="*60)
        print("🛡️  WEAPON DETECTION SYSTEM STARTED")
        print("="*60)
        print("Options:")
        print("  1. Upload Image - Detect weapons in a single image")
        print("  2. Upload Video - Process video file")
        print("  3. Start Webcam - Real-time detection")
        print("\nAdjust confidence threshold using the slider")
        print("Detections are logged to: weapon_detections/detection_log.txt")
        print("="*60 + "\n")
        
        self.root.mainloop()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Path to your trained model
    MODEL_PATH = r"C:\Users\Jiya\OneDrive - somaiya.edu\Documents\LY Project\final run\weapon_detection_single_class\weights\best.pt"  # Change this to your model path
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found!")
        print(f"Please place your best.pt file at: {MODEL_PATH}")
        print("\nOr update MODEL_PATH in the script to point to your model location.")
        input("\nPress Enter to exit...")
        exit(1)
    
    try:
        # Create and run app
        app = WeaponDetectionApp(MODEL_PATH)
        app.run()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")