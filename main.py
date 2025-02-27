import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import time
import os
import pytesseract
from pytesseract import Output
import re
import winsound  # For sound alert on Windows

# Ensure 'yolov5' folder exists for model storage
os.makedirs("yolov5", exist_ok=True)

# Load YOLOv5 model (Pretrained on COCO dataset)
model_path = "yolov5/yolov5s.pt"

if not os.path.exists(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    torch.save(model.state_dict(), model_path)
else:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def apply_blur(frame, x1, y1, x2, y2, blur_type="gaussian"):
    """Applies adaptive blurring intensity with finer control based on object size."""
    roi = frame[y1:y2, x1:x2]
    obj_width = x2 - x1
    obj_height = y2 - y1
    obj_area = obj_width * obj_height

    # Dynamic blur intensity calculation (smooth scaling)
    blur_strength = max(15, min(60, 60 - obj_area // 1500))
    blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1  # Ensure it's odd for GaussianBlur

    if blur_type == "gaussian":
        ksize = (blur_strength, blur_strength)
        blurred = cv2.GaussianBlur(roi, ksize, 30)

    elif blur_type == "pixelation":
        h, w = roi.shape[:2]
        temp = cv2.resize(roi, (w // (blur_strength // 5), h // (blur_strength // 5)), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    elif blur_type == "mosaic":
        h, w = roi.shape[:2]
        temp = cv2.resize(roi, (w // (blur_strength // 7), h // (blur_strength // 7)), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    else:
        return roi

    frame[y1:y2, x1:x2] = blurred
    return frame


def is_sensitive_text(text):
    """Checks if a text is considered sensitive (e.g., emails, passwords)."""
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    return re.search(email_pattern, text) is not None


def blur_text_regions(frame, blur_type="gaussian"):
    """Detects text using OCR and applies blurring to sensitive words."""
    ocr_data = pytesseract.image_to_data(frame, output_type=Output.DICT)

    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i]
        if text.strip() and is_sensitive_text(text):  # Check for sensitive text
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                          ocr_data['width'][i], ocr_data['height'][i])
            frame = apply_blur(frame, x, y, x + w, y + h, blur_type)
    return frame


def detect_and_blur(frame, blur_type="gaussian", stealth_mode=False):
    """Detects sensitive objects and applies the selected blurring technique."""
    results = model(frame)
    security_enabled = False

    if not stealth_mode:
        # Draw bounding boxes and labels only if stealth mode is off
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            if cls in [67, 73]:  # Add appropriate class IDs for security objects
                frame = apply_blur(frame, x1, y1, x2, y2, blur_type)
                security_enabled = True

            color = (0, 255, 0) if cls not in [67, 73] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Apply OCR-based text blurring regardless of stealth mode
    frame = blur_text_regions(frame, blur_type)

    return frame, security_enabled


def play_alert_sound():
    """Plays an alert beep sound when security is triggered."""
    winsound.Beep(1000, 500)  # 1000Hz frequency for 500ms


class SecurityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Security System")
        self.root.geometry("380x520")
        self.root.configure(bg="#2c3e50")

        self.blur_type = tk.StringVar(value="gaussian")  # Default blur type
        self.stealth_mode = False  # Track stealth mode state

        btn_style = {
            "font": ("Arial", 12, "bold"),
            "bg": "#3498db",
            "fg": "white",
            "width": 18,
            "height": 2,
            "bd": 3,
            "relief": "raised"
        }

        self.btn_webcam = tk.Button(root, text="Start Webcam", command=self.start_webcam, **btn_style)
        self.btn_stop_webcam = tk.Button(root, text="Stop Webcam", command=self.stop_webcam, **btn_style)
        self.btn_video = tk.Button(root, text="Load Video", command=self.load_video, **btn_style)
        self.btn_toggle_stealth = tk.Button(root, text="Toggle Stealth Mode", command=self.toggle_stealth_mode, **btn_style)
        self.btn_exit = tk.Button(root, text="Exit", command=self.exit_app, **btn_style)

        # Blur type selection (Radio Buttons)
        tk.Label(root, text="Select Blur Type:", font=("Arial", 12, "bold"), bg="#2c3e50", fg="white").pack(pady=5)

        blur_frame = tk.Frame(root, bg="#2c3e50")
        blur_frame.pack()

        self.radio_gaussian = tk.Radiobutton(blur_frame, text="Gaussian Blur", variable=self.blur_type, value="gaussian",
                                             font=("Arial", 10), bg="#2c3e50", fg="white", selectcolor="#e74c3c", indicatoron=True)
        self.radio_pixelation = tk.Radiobutton(blur_frame, text="Pixelation", variable=self.blur_type, value="pixelation",
                                               font=("Arial", 10), bg="#2c3e50", fg="white", selectcolor="#e74c3c", indicatoron=True)
        self.radio_mosaic = tk.Radiobutton(blur_frame, text="Mosaic", variable=self.blur_type, value="mosaic",
                                           font=("Arial", 10), bg="#2c3e50", fg="white", selectcolor="#e74c3c", indicatoron=True)

        self.radio_gaussian.pack(anchor="w", pady=2)
        self.radio_pixelation.pack(anchor="w", pady=2)
        self.radio_mosaic.pack(anchor="w", pady=2)

        self.btn_webcam.pack(pady=10)
        self.btn_stop_webcam.pack(pady=10)
        self.btn_video.pack(pady=10)
        self.btn_toggle_stealth.pack(pady=10)
        self.btn_exit.pack(pady=10)

        self.capture = None
        self.running = False

    def start_webcam(self):
        """Starts the webcam feed for real-time detection."""
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

    def toggle_stealth_mode(self):
        """Toggles stealth mode on and off."""
        self.stealth_mode = not self.stealth_mode
        status = "enabled" if self.stealth_mode else "disabled"
        self.btn_toggle_stealth.config(text=f"Stealth Mode {status}")
        print(f"Stealth mode {status}")

    def process_feed(self):
        """Processes video/webcam feed, applies YOLOv5 detection, and shows FPS and security status."""
        prev_time = time.time()
        while self.running and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                break

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            frame, security_enabled = detect_and_blur(frame, self.blur_type.get(), stealth_mode=self.stealth_mode)

            cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            security_text = "Security Enabled" if security_enabled else "No Need of Security"
            color = (0, 0, 255) if security_enabled else (0, 255, 0)
            cv2.putText(frame, security_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if security_enabled:
                play_alert_sound()  # Play the beep sound when security is triggered

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
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityApp(root)
    root.mainloop()
