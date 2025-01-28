import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import time
import os

# Ensure 'yolov5' folder exists for model storage
os.makedirs("yolov5", exist_ok=True)

# Load YOLOv5 model (Pretrained on COCO dataset)
model_path = "yolov5/yolov5s.pt"

# Check if the model exists; otherwise, let torch.hub.download it
if not os.path.exists(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    torch.save(model.state_dict(), model_path)  # Save model inside yolov5 folder
else:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load normally


def detect_and_blur(frame):
    """Runs YOLOv5 detection, draws bounding boxes, and blurs sensitive objects."""
    results = model(frame)
    security_enabled = False

    for det in results.xyxy[0]:  # Bounding box format: (x1, y1, x2, y2, conf, class)
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        if cls in [67, 73]:  # 67 = Cell phone, 73 = Laptop
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (25, 25), 30)
            security_enabled = True

        # Draw bounding box
        color = (0, 255, 0) if cls not in [67, 73] else (0, 0, 255)  # Red for sensitive objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame, security_enabled

class SecurityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Security System")
        self.root.geometry("350x450")  # Increased window size
        self.root.configure(bg="#2c3e50")  # Set background color

        # Styling for buttons
        btn_style = {
            "font": ("Arial", 14, "bold"),
            "bg": "#3498db",
            "fg": "white",
            "width": 18,
            "height": 2,
            "bd": 4,
            "relief": "raised"
        }

        # Buttons to switch input sources
        self.btn_webcam = tk.Button(root, text="Start Webcam", command=self.start_webcam, **btn_style)
        self.btn_stop_webcam = tk.Button(root, text="Stop Webcam", command=self.stop_webcam, **btn_style)
        self.btn_video = tk.Button(root, text="Load Video", command=self.load_video, **btn_style)
        self.btn_exit = tk.Button(root, text="Exit", command=self.exit_app, **btn_style)

        # Arrange buttons with better spacing
        self.btn_webcam.pack(pady=20)
        self.btn_stop_webcam.pack(pady=20)
        self.btn_video.pack(pady=20)
        self.btn_exit.pack(pady=20)

        self.capture = None
        self.running = False

    def start_webcam(self):
        """Starts webcam feed for real-time detection."""
        if self.capture is not None:
            self.stop_webcam()

        self.running = True
        self.capture = cv2.VideoCapture(0)
        Thread(target=self.process_feed).start()

    def stop_webcam(self):
        """Stops the webcam feed without exiting the program."""
        self.running = False
        if self.capture:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()

    def load_video(self):
        """Loads a video file for processing."""
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            if self.capture is not None:
                self.stop_webcam()

            self.running = True
            self.capture = cv2.VideoCapture(file_path)
            Thread(target=self.process_feed).start()

    def process_feed(self):
        """Processes video/webcam feed, applies YOLOv5 detection, and shows FPS and security status."""
        prev_time = time.time()
        while self.running and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            curr_time = time.time()
            time_diff = curr_time - prev_time
            prev_time = curr_time  # Update time before FPS calculation

            fps = int(1 / time_diff) if time_diff > 0 else 0  # Prevent division by zero

            frame, security_enabled = detect_and_blur(frame)

            # Display FPS on frame in red
            cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display security status
            security_text = "Security Enabled" if security_enabled else "No Need of Security"
            color = (0, 0, 255) if security_enabled else (0, 255, 0)
            cv2.putText(frame, security_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("AI Security System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop_webcam()

    def exit_app(self):
        """Exits the application and releases resources."""
        self.running = False
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
        self.root.quit()
    
if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityApp(root)
    root.mainloop()
