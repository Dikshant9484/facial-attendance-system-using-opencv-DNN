🎓 Face Recognition Attendance System

This project is an AI-powered attendance system that uses DeepFace (FaceNet model) and OpenCV to automatically detect and verify student faces through a webcam. When a student is recognized, their attendance is logged in a daily CSV file.

⚡ Key Features:

Real-time face detection using OpenCV.

Face verification with DeepFace (Facenet model).

Marks attendance only once per student within a cooldown period (default: 60s).

Logs stored in CSV format with timestamp.

Easy to extend with new student images.

.
├── data/
│   └── students/         # Student images (NOT included in repo)
├── logs/                 # Attendance CSV logs (auto-generated)
├── attendance.py         # Main Python script
└── README.md             # Documentation



🔒 Note on Dataset

The student dataset (images) is NOT included in this repository due to privacy and security concerns. You must add your own images in the data/students/ folder for the system to work.

🚀 Future Improvements

Add GUI for easier management.

Support for multiple face recognition backends (VGG-Face, ArcFace, etc.).

Integration with databases instead of CSV.

Cloud-based attendance dashboard.
