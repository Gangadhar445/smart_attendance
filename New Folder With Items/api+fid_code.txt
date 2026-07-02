import face_recognition
import cv2
import os
import numpy as np
import csv
from datetime import datetime
import threading

# --- FastAPI Imports ---
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd

# --- 1. Load dataset and create face encodings ---
dataset_path = 'dataset'
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

# --- 2. Attendance Logging ---
attendance_file = 'attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Date', 'Time'])

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    existing_entries = []
    if os.path.exists(attendance_file):
        with open(attendance_file, 'r') as f:
            reader = csv.reader(f)
            existing_entries = list(reader)

    for row in existing_entries:
        if len(row) < 2:
            continue
        if row[0] == name and row[1] == date_str:
            return

    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
        print(f"Attendance marked for {name} at {time_str}")

# --- 3. FastAPI Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

def load_attendance():
    try:
        df = pd.read_csv(attendance_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return pd.DataFrame(columns=['Name', 'Date', 'Time'])

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard_realtime.html", {"request": request})

@app.get("/attendance")
def attendance_data():
    df = load_attendance()
    total_students = df['Name'].nunique()
    today = pd.Timestamp.now().normalize()
    today_present_students = df[df['Date'] == today]['Name'].unique()
    today_attendance = len(today_present_students)
    today_absent = total_students - today_attendance

    # Attendance % per student
    attendance_percentage = {
        student: len(df[df['Name'] == student]) / len(df['Date'].unique()) * 100
        for student in df['Name'].unique()
    }

    # Top Engaged Students as table data
    engagement = df['Name'].value_counts().reset_index()
    engagement.columns = ['Name', 'Attendance Count']
    engage_json = {
        "students": engagement['Name'].tolist(),
        "counts": engagement['Attendance Count'].tolist(),
        "total_days": len(df['Date'].unique())
    }

    # Heatmap matrix
    heatmap_students = sorted(df['Name'].unique())
    heatmap_dates = sorted(df['Date'].dt.strftime('%Y-%m-%d').unique())
    heatmap_matrix = []
    for student in heatmap_students:
        row = []
        for d in heatmap_dates:
            present = not df[(df['Name'] == student) & (df['Date'].dt.strftime('%Y-%m-%d') == d)].empty
            row.append(1 if present else 0)
        heatmap_matrix.append(row)

    return JSONResponse({
        "total_students": total_students,
        "today_attendance": today_attendance,
        "today_absent": today_absent,
        "attendance_percentage": attendance_percentage,
        "engage_json": engage_json,
        "heatmap_matrix": heatmap_matrix,
        "heatmap_students": heatmap_students,
        "heatmap_dates": heatmap_dates
    })

def run_dashboard():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# --- 4. Face Recognition Loop ---
def face_recognition_loop():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(student_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(student_encodings, face_encoding)

            name = "Unknown"
            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = student_names[best_match_index]
                mark_attendance(name)

            # Draw rectangle and label
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Attendance System', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 5. Run Both Threads ---
if __name__ == "__main__":
    threading.Thread(target=run_dashboard, daemon=True).start()
    face_recognition_loop()
