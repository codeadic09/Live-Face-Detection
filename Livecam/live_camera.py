import face_recognition
import cv2
import pickle
import numpy as np
import os

encodings_file = 'encodings.pkl'
tolerance = 2.0  # For accuracy (lower = stricter)

# Load enrolled faces (optional - works without)
known_encodings = {}
known_names = []
if os.path.exists(encodings_file):
    with open(encodings_file, 'rb') as f:
        known_encodings = pickle.load(f)
    known_names = list(known_encodings.keys())
    print(f"✅ Loaded {len(known_names)} people")

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera error! Try index 1 or 2.")
    exit()

print("✅ Camera on! Live scanning all faces. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize for faster processing (keeps it real-time)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect all faces (fast 'hog' model for real-time)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Recognize each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back to original size
        top *= 2; right *= 2; bottom *= 2; left *= 2

        name = "Unknown"
        accuracy = 0
        if known_encodings:
            matches = face_recognition.compare_faces(list(known_encodings.values()), face_encoding, tolerance)
            distances = face_recognition.face_distance(list(known_encodings.values()), face_encoding)
            if len(distances) > 0:
                best_index = np.argmin(distances)
                if matches[best_index]:
                    name = known_names[best_index]
                    accuracy = (1 - distances[best_index]) * 100

        # Draw box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} ({accuracy:.1f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Show live updated frame
    cv2.imshow('Live Face Scan - Press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Camera off")
