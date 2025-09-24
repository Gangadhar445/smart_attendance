import face_recognition
import cv2
import os
import numpy as np
import csv
from datetime import datetime
from tabulate import tabulate
import pandas as pd

# --- 1. Paths and Initialization ---
dataset_path = 'dataset'
attendance_file = 'attendance.csv'
student_encodings = []
student_names = []

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load existing attendance file or create it
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Date', 'Time'])

# --- 2. Load dataset and create encodings ---
def load_dataset():
    global student_encodings, student_names
    student_encodings = []
    student_names = []
    for student_name in os.listdir(dataset_path):
        student_folder = os.path.join(dataset_path, student_name)
        if not os.path.isdir(student_folder):
            continue
        for img_file in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                student_encodings.append(encodings[0])
                student_names.append(student_name)
    print(f"Loaded encodings for {len(student_names)} images.")

load_dataset()

# --- 3. Attendance marking ---
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Avoid duplicate attendance for the same day
    existing_entries = []
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            existing_entries = list(reader)
    
    for row in existing_entries[1:]:  # skip header
        if len(row) >= 2 and row[0] == name and row[1] == date_str:
            return

    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
        print(f"Attendance marked for {name} at {time_str}")

# --- 4. Register new student ---
def register_student():
    name = input("Enter student name: ").strip()
    student_folder = os.path.join(dataset_path, name)
    if not os.path.exists(student_folder):
        os.makedirs(student_folder)
    cap = cv2.VideoCapture(0)
    print("Capturing 5 snapshots. Please look at the camera...")
    count = 0
    while count < 5:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Register Student - Press 'c' to capture", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            img_path = os.path.join(student_folder, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved snapshot {count + 1}")
            count += 1
    cap.release()
    cv2.destroyAllWindows()
    load_dataset()
    print(f"Student {name} registered successfully!")

# --- 5. Initialize webcam ---
cap = cv2.VideoCapture(0)

print("Starting Attendance System. Press 'r' to register a new student, 'q' to quit.")

# --- 6. Face recognition loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    unknown_present = False  # flag to detect unknown faces

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(student_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(student_encodings, face_encoding)

        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = student_names[best_match_index]
            mark_attendance(name)
        else:
            unknown_present = True  # set flag if unknown face detected

        # Draw rectangle and label
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Attendance System', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        if unknown_present:
            register_student()
        else:
            print("No unknown face detected. Cannot register a student.")

cap.release()
cv2.destroyAllWindows()

# --- 7. Print attendance in structured wide table ---
df = pd.read_csv(attendance_file)
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# Pivot table: one row per student, columns = dates
pivot_df = df.pivot_table(index='Name', columns='Date', values='Time', aggfunc='first').fillna('')
pivot_df.reset_index(inplace=True)

print("\nAttendance Table:")
print(tabulate(pivot_df, headers=pivot_df.columns, tablefmt="grid"))
