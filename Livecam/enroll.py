import face_recognition
import pickle
import os
import numpy as np

enroll_folder = 'enroll'
encodings_file = 'encodings.pkl'

encodings = {}
for person in os.listdir(enroll_folder):
    person_dir = os.path.join(enroll_folder, person)
    person_encodings = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = face_recognition.load_image_file(img_path)
        enc = face_recognition.face_encodings(image)
        if enc:
            person_encodings.append(enc[0])
    if person_encodings:
        encodings[person] = np.mean(person_encodings, axis=0)

with open(encodings_file, 'wb') as f:
    pickle.dump(encodings, f)

print("✅ Enrollment done!")
