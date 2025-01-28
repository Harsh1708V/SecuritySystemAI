# AI Security System

## 📌 Project Overview
The **AI Security System** is a computer vision-based application that detects and blurs sensitive objects (such as mobile phones and laptops) in real-time video feeds. It uses **YOLOv5** for object detection and supports both webcam and video input sources. The project is built with **Python, OpenCV, and Tkinter**.

## 📂 Project Structure
```
AI-Security-System/
│── yolov5/                # Folder to store YOLOv5 model
│    └── yolov5s.pt        # Pretrained YOLOv5 small model
│── main.py                # Main application script
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── LICENSE                # License file
│── config/                # Configuration files (if needed)
│── utils/                 # Helper functions (if needed)
```

## 🚀 Features
✅ **Real-time object detection** using YOLOv5  
✅ **Blurs sensitive information** (phones & laptops)  
✅ **Switchable input sources** (Webcam & Video)  
✅ **Live FPS Display** for performance tracking  
✅ **Interactive Tkinter GUI** for easy control  

## 🛠 Installation & Setup
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/AI-Security-System.git
cd AI-Security-System
```

### **2️⃣ Set Up Virtual Environment (Recommended)**
```sh
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### **3️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **4️⃣ Download YOLOv5 Model**
The script will automatically download the model (`yolov5s.pt`). If needed, you can manually place it in the `yolov5/` directory.

```sh
mkdir yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -P yolov5/
```

## ▶️ Running the Application
```sh
python main.py
```

## 🏗 How It Works
1. **Select an Input Source** → Use buttons to start webcam or load a video file.
2. **YOLOv5 Object Detection** → Detects objects in real-time.
3. **Blurs Sensitive Objects** → Phones & laptops are blurred for privacy.
4. **Displays FPS & Security Status** → Helps monitor performance.

## 🌍 Contributing
We welcome contributions! To contribute:
1. **Fork** the repository.
2. **Create a new branch** (`feature-branch`)
3. **Commit your changes**
4. **Push to your fork**
5. **Submit a Pull Request (PR)**

## ⚖️ License
This project is licensed under the **MIT License**. Feel free to modify and use it!

## 📧 Contact
For queries or support, contact [E-Mail](mailto:harshwaibhav69@gmail.com).

---
**Happy Coding! 🚀**
