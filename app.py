import os
import cv2
import time
import csv
import datetime as dt
from deepface import DeepFace

STUDENTS_DIR = "data/students"
LOG_DIR = "logs"
TOLERANCE = 0.4         
DEDUPE_COOLDOWN_S = 60

os.makedirs(LOG_DIR, exist_ok=True)

print("[+] Loading student images...")
KNOWN_FACES = {}  
for fn in os.listdir(STUDENTS_DIR):
    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(fn)[0]
        KNOWN_FACES[name] = os.path.join(STUDENTS_DIR, fn)
print(f"[✓] Loaded {len(KNOWN_FACES)} students.")


last_marked = {}

# CSV for today
today = dt.date.today().isoformat()
csv_path = os.path.join(LOG_DIR, f"attendance_{today}.csv")
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "name", "status"])

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found")

print("[i] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save temp image for checking
    cv2.imwrite("temp.jpg", frame)

    try:
        # Detect and analyze faces
        faces = DeepFace.extract_faces("temp.jpg", detector_backend="opencv")
        for face in faces:
            x, y, w, h = face["facial_area"].values()
            face_img = frame[y:y+h, x:x+w]

            name = "Unknown"
            best_score = 1.0

            
            for student_name, student_img in KNOWN_FACES.items():
                result = DeepFace.verify(face_img, student_img, model_name="Facenet", enforce_detection=False)
                score = result["distance"]
                if result["verified"] and score < TOLERANCE and score < best_score:
                    name = student_name
                    best_score = score


            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            
            if name != "Unknown":
                now = time.time()
                last_time = last_marked.get(name, 0)
                if now - last_time > DEDUPE_COOLDOWN_S:
                    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([ts, name, "Present"])
                    last_marked[name] = now
                    print(f"[✓] Marked {name} at {ts}")

    except Exception as e:
        pass  

    # Show frame
    cv2.imshow("Attendance (press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[✓] Attendance saved to {csv_path}")
