"""
attend.py

Face Recognition Attendance System (with interval CSV persistence and
final attendance marking only when student meets interval threshold).

Behavior:
- INTERVAL_FILE stores per-session interval hits immediately.
- Student is appended to ATTENDANCE_FILE (one row per day) ONLY when
  they have interval hits >= session threshold (e.g. 3 of 5).
"""
import os
import cv2
import csv
import time
import json
import difflib
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import threading
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# -------------------------
# Config / Paths
# -------------------------
DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"             # final daily attendance rows (one per student/day)
INTERVAL_FILE = "interval_attendance.csv"      # per-session per-interval hits
TEMPLATES_DIR = "templates"
ENCODING_MODEL = "hog"                      # 'hog' (CPU) or 'cnn' (GPU/dlib-cuda)
CAMERA_NAMES_FILE = "camera_names.json"
PROBE_MAX = 6                               # probe indices 0..PROBE_MAX-1

# Class session defaults
SESSIONS_FILE = "sessions.json"
CLASS_DURATION_MIN = 50
INTERVAL_MIN = 10

# Ensure dataset and CSVs exist
os.makedirs(DATASET_PATH, exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

if not os.path.exists(INTERVAL_FILE):
    with open(INTERVAL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "name", "interval_index", "timestamp_utc"])

# -------------------------
# Sessions store (interval monitoring)
# -------------------------
_sessions_lock = threading.Lock()

def load_sessions():
    try:
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_sessions(sessions):
    try:
        with open(SESSIONS_FILE, "w") as f:
            json.dump(sessions, f, indent=2, default=str)
    except Exception as e:
        print("[SESSIONS] save failed:", e)

SESSIONS = load_sessions()

def create_session(start_time=None, duration_min=CLASS_DURATION_MIN, interval_min=INTERVAL_MIN):
    with _sessions_lock:
        sid = str(int(time.time()*1000))
        st = datetime.utcnow() if start_time is None else (datetime.fromisoformat(start_time) if isinstance(start_time, str) else start_time)
        intervals = max(1, duration_min // interval_min)
        SESSIONS[sid] = {
            "start_iso": st.isoformat(),
            "duration_min": duration_min,
            "interval_min": interval_min,
            "intervals": intervals,
            "active": True,
            "attendance": {}  # mapping name -> list of interval indices (in-memory)
        }
        save_sessions(SESSIONS)
        print(f"[SESSIONS] created {sid} starting {st.isoformat()} intervals={intervals}")
        return sid

def stop_session(session_id):
    with _sessions_lock:
        sess = SESSIONS.get(session_id)
        if sess:
            sess["active"] = False
            save_sessions(SESSIONS)
            return True
    return False

def get_active_session():
    with _sessions_lock:
        for sid, s in SESSIONS.items():
            if s.get("active"):
                return sid, s
    return None, None

def compute_interval_index_for_session(sess, dt_utc):
    start = datetime.fromisoformat(sess["start_iso"])
    delta = dt_utc - start
    if delta.total_seconds() < 0:
        return None
    dur_seconds = sess["duration_min"] * 60
    if delta.total_seconds() >= dur_seconds:
        return None
    idx = int(delta.total_seconds() // (sess["interval_min"] * 60))
    if idx < 0 or idx >= sess["intervals"]:
        return None
    return idx

# -------------------------
# Interval CSV utilities
# -------------------------
def interval_csv_has(session_id, name, interval_index):
    """Return True if CSV already contains the (session_id,name,interval_index) triplet."""
    try:
        with open(INTERVAL_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] == session_id and row[1] == name and str(row[2]) == str(interval_index):
                    return True
    except Exception:
        pass
    return False

def append_interval_csv(session_id, name, interval_index, timestamp_utc):
    try:
        with open(INTERVAL_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([session_id, name, interval_index, timestamp_utc.isoformat()])
    except Exception as e:
        print("[INTERVAL CSV] write failed:", e)

# -------------------------
# Attendance (daily CSV) helpers
# -------------------------
def already_marked_today(name):
    """Return True if ATTENDANCE_FILE already has name for today's date."""
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    return True
    except Exception:
        pass
    return False

def append_attendance_now(name, when_local=None):
    """Append a row to ATTENDANCE_FILE for name with the given local timestamp."""
    try:
        now = when_local or datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        # Double-check not duplicate
        if already_marked_today(name):
            return False
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        print(f"[FINAL MARK] {name} on {date_str} at {time_str}")
        return True
    except Exception as e:
        print("[ATTENDANCE CSV] write failed:", e)
        return False

# -------------------------
# Add interval presence and check threshold -> final attendance
# -------------------------
def add_interval_presence(name, timestamp_utc):
    """
    Record presence for active session in both in-memory SESSIONS and the INTERVAL_FILE.
    After appending interval record, check if the student's interval-count >= threshold;
    if so, append final attendance row to ATTENDANCE_FILE (if not already present).
    Returns the interval index or None.
    """
    sid, sess = get_active_session()
    if not sid:
        return None
    idx = compute_interval_index_for_session(sess, timestamp_utc)
    if idx is None:
        return None
    # write to INTERVAL_FILE if not duplicate
    if not interval_csv_has(sid, name, idx):
        append_interval_csv(sid, name, idx, timestamp_utc)
    # update in-memory sessions map
    with _sessions_lock:
        attendance = sess.setdefault("attendance", {})
        arr = attendance.setdefault(name, [])
        if idx not in arr:
            arr.append(idx)
        SESSIONS[sid] = sess
        save_sessions(SESSIONS)
    # Now check threshold and mark final attendance if threshold met
    # Determine interval count for this student for this session (merge in-memory + CSV)
    indices_set = set(sess.get("attendance", {}).get(name, []))
    # include rows from INTERVAL_FILE for this session & name (in case of restart)
    try:
        with open(INTERVAL_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] == sid and row[1] == name:
                    try:
                        indices_set.add(int(row[2]))
                    except Exception:
                        pass
    except Exception:
        pass
    count = len(indices_set)
    intervals_total = sess.get("intervals", max(1, sess.get("duration_min", CLASS_DURATION_MIN)//sess.get("interval_min", INTERVAL_MIN)))
    # threshold policy: default is ceil(60%) (as earlier).
    threshold = max(1, int((intervals_total * 0.6) + 0.9999))
    if count >= threshold:
        # If not already marked in ATTENDANCE_FILE for today, append
        if not already_marked_today(name):
            append_attendance_now(name, when_local=datetime.now())
            return idx
    return idx

# -------------------------
# Dataset load & earlier mark_attendance function removed from recognition path
# (we use add_interval_presence instead)
# -------------------------
student_encodings = []
student_names = []

def load_dataset():
    global student_encodings, student_names
    student_encodings = []
    student_names = []
    if not os.path.exists(DATASET_PATH):
        print(f"[WARN] Dataset folder '{DATASET_PATH}' missing.")
        return
    for person in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_folder):
            continue
        for fname in os.listdir(person_folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_folder, fname)
            try:
                image = face_recognition.load_image_file(img_path)
                boxes = face_recognition.face_locations(image, model=ENCODING_MODEL)
                encs = face_recognition.face_encodings(image, boxes)
                if encs:
                    student_encodings.append(encs[0])
                    student_names.append(person)
            except Exception as e:
                print(f"[WARN] Failed {img_path}: {e}")
    print(f"[INFO] Loaded {len(student_names)} known face images.")

# Initial dataset load
load_dataset()

# -------------------------
# CameraManager (manual reprobe)
# -------------------------
class CameraManager:
    def __init__(self, start_index=0, probe_max=PROBE_MAX):
        self.lock = threading.Lock()
        self.index = int(start_index)
        self.capture = None
        self.probe_max = probe_max
        self.available = self.probe_cameras()
        self.names = self.load_camera_names()
        for idx in self.available:
            if str(idx) not in self.names:
                self.names[str(idx)] = f"Camera {idx}"
        if self.index not in self.available:
            self.index = self.available[0] if self.available else 0
        self.open(self.index)
        self.save_camera_names()

    def probe_cameras(self):
        found = []
        for i in range(self.probe_max):
            cap = cv2.VideoCapture(i)
            time.sleep(0.12)
            ok, _ = cap.read()
            try: cap.release()
            except Exception: pass
            if ok:
                found.append(i)
        print(f"[CAM] Probed: {found}")
        return found

    def load_camera_names(self):
        if os.path.exists(CAMERA_NAMES_FILE):
            try:
                with open(CAMERA_NAMES_FILE, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def save_camera_names(self):
        try:
            with open(CAMERA_NAMES_FILE, "w") as f:
                json.dump(self.names, f, indent=2)
        except Exception as e:
            print("[CAM] save names failed:", e)

    def open(self, index):
        index = int(index)
        with self.lock:
            try:
                if self.capture is not None:
                    try: self.capture.release()
                    except Exception: pass
            except Exception:
                pass
            cap = cv2.VideoCapture(index)
            time.sleep(0.2)
            if not cap.isOpened():
                print(f"[CAM] Unable to open {index}")
                if self.available:
                    fallback = self.available[0]
                    if fallback != index:
                        cap0 = cv2.VideoCapture(fallback)
                        time.sleep(0.2)
                        if cap0.isOpened():
                            self.capture = cap0
                            self.index = fallback
                            return True
                self.capture = None
                return False
            else:
                self.capture = cap
                self.index = index
                if index not in self.available:
                    self.available.append(index)
                print(f"[CAM] Switched to {index}")
                return True

    def read(self):
        with self.lock:
            if self.capture is None:
                return False, None
            ret, frame = self.capture.read()
            return ret, frame

    def release(self):
        with self.lock:
            try:
                if self.capture is not None:
                    self.capture.release()
            except Exception:
                pass
            self.capture = None

    def get_index(self):
        with self.lock:
            return self.index

    def get_available(self):
        return list(self.available)

    def get_name(self, index):
        return self.names.get(str(index), f"Camera {index}")

    def reprobe(self):
        with self.lock:
            new_found = []
            for i in range(self.probe_max):
                cap = cv2.VideoCapture(i)
                time.sleep(0.12)
                ok, _ = cap.read()
                try: cap.release()
                except Exception: pass
                if ok: new_found.append(i)
            self.available = new_found
            for idx in self.available:
                if str(idx) not in self.names:
                    self.names[str(idx)] = f"Camera {idx}"
            if self.index not in self.available:
                if self.available:
                    first = self.available[0]
                    try:
                        self.open(first)
                    except Exception as e:
                        print("[CAM] reprobe open failed:", e)
                else:
                    print("[CAM] No cameras after reprobe.")
                    try:
                        if self.capture is not None:
                            self.capture.release()
                    except Exception:
                        pass
                    self.capture = None
            self.save_camera_names()
            return list(self.available)

CAM = CameraManager(start_index=0, probe_max=PROBE_MAX)

# -------------------------
# Registration (auto-capture via CAM)
# -------------------------
def normalize_name(s: str) -> str:
    return "".join(s.lower().split())

def get_existing_names():
    if not os.path.exists(DATASET_PATH):
        return []
    return [n for n in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, n))]

def register_student_interactive():
    existing_names = get_existing_names()
    existing_norm = {normalize_name(n): n for n in existing_names}
    while True:
        name = input("Enter student name: ").strip()
        if not name:
            print("[INFO] Name empty. Aborting registration.")
            return
        if normalize_name(name) in existing_norm:
            print(f"[WARN] A similar/exact name already exists: '{existing_norm[normalize_name(name)]}'")
            choice = input("Enter 'r' to retry or 'a' to abort: ").strip().lower()
            if choice == "r":
                continue
            else:
                print("[INFO] Registration aborted.")
                return
        suggestions = difflib.get_close_matches(name, existing_names, n=3, cutoff=0.75)
        if suggestions:
            print("[WARN] Similar names found:")
            for s in suggestions: print("  -", s)
            confirm = input("Proceed with this name anyway? (y/N) ").strip().lower()
            if confirm != "y":
                print("[INFO] Enter a different name.")
                continue
        break

    person_dir = os.path.join(DATASET_PATH, name)
    os.makedirs(person_dir, exist_ok=True)
    max_shots = 5
    delay_between = 0.6
    capturing = False
    captured = 0
    print("[INFO] Registration preview started using camera index", CAM.get_index())
    print(f" - Press 'c' once to start auto-capture of {max_shots} images. Press 'q' to cancel.")
    try:
        while True:
            ret, frame = CAM.read()
            if not ret:
                time.sleep(0.05)
                continue
            disp = frame.copy()
            if not capturing:
                cv2.putText(disp, "Press 'c' to START auto-capture | 'q' to cancel", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            else:
                cv2.putText(disp, f"Capturing... {captured}/{max_shots}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Register Student (auto-capture)", disp)
            key = cv2.waitKey(1) & 0xFF
            if not capturing and key == ord("c"):
                capturing = True
                captured = 0
                print("[INFO] Auto-capture started...")
                while captured < max_shots:
                    ret2, frame2 = CAM.read()
                    if not ret2:
                        time.sleep(0.02)
                        continue
                    timestamp = int(time.time() * 1000)
                    fname = f"{name}_{timestamp}_{captured}.jpg"
                    out_path = os.path.join(person_dir, fname)
                    cv2.imwrite(out_path, frame2)
                    captured += 1
                    print(f"[SAVED] {captured}/{max_shots}: {out_path}")
                    waited_ms = 0
                    wait_ms_total = int(delay_between * 1000)
                    while waited_ms < wait_ms_total:
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("q"):
                            print("[INFO] Registration cancelled during auto-capture.")
                            capturing = False
                            break
                        time.sleep(0.01)
                        waited_ms += 10
                    if not capturing:
                        break
                print("[INFO] Auto-capture finished." if captured >= max_shots else "[INFO] Auto-capture stopped early.")
                break
            elif key == ord("q"):
                print("[INFO] Registration cancelled by user.")
                break
    finally:
        cv2.destroyAllWindows()

    if os.path.exists(person_dir) and len(os.listdir(person_dir)) > 0:
        load_dataset()
        print(f"[INFO] Registered '{name}' with {len(os.listdir(person_dir))} images.")
    else:
        print("[INFO] No images saved. Registration aborted or incomplete.")

# -------------------------
# FastAPI + endpoints
# -------------------------
app = FastAPI(title="Attendance Dashboard")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def load_attendance_df():
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return pd.DataFrame(columns=['Name','Date','Time'])

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    template_file = os.path.join(TEMPLATES_DIR, "dashboard_realtime.html")
    if os.path.exists(template_file):
        return templates.TemplateResponse("dashboard_realtime.html", {"request": request})
    df = load_attendance_df()
    if df.empty:
        html = "<h3>No attendance data.</h3>"
    else:
        df2 = df.copy(); df2['Date'] = df2['Date'].dt.strftime('%Y-%m-%d'); html = df2.to_html(index=False)
    return HTMLResponse(f"<html><body>{html}</body></html>")

@app.get("/attendance")
def attendance_data():
    df = load_attendance_df()
    if df.empty:
        return JSONResponse({
            "total_students": 0,
            "today_attendance": 0,
            "today_absent": 0,
            "attendance_percentage": {},
            "engage_json": {"students":[], "counts":[], "total_days":0},
            "heatmap_matrix": [], "heatmap_students": [], "heatmap_dates": []
        })
    total_students = int(df['Name'].nunique())
    df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    today_present_students = df[df['Date_str'] == today_str]['Name'].unique().tolist()
    today_attendance = len(today_present_students)
    today_absent = max(total_students - today_attendance, 0)
    unique_dates = sorted(df['Date_str'].unique())
    attendance_percentage = {}
    for student in df['Name'].unique():
        count = len(df[df['Name'] == student])
        attendance_percentage[student] = (count / len(unique_dates)) * 100 if len(unique_dates) > 0 else 0
    engagement = df['Name'].value_counts().reset_index()
    engagement.columns = ['Name', 'Attendance Count']
    engage_json = {
        "students": engagement['Name'].tolist(),
        "counts": engagement['Attendance Count'].tolist(),
        "total_days": len(unique_dates)
    }
    heatmap_students = sorted(df['Name'].unique().tolist())
    heatmap_dates = unique_dates
    heatmap_matrix = []
    for student in heatmap_students:
        row = []
        for d in heatmap_dates:
            present = not df[(df['Name'] == student) & (df['Date_str'] == d)].empty
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

# cameras endpoints
@app.get("/cameras")
def cameras_list():
    cams = CAM.get_available()
    out = [{"index": i, "name": CAM.get_name(i)} for i in cams]
    return JSONResponse({"cameras": out, "current": CAM.get_index()})

@app.post("/camera/{index}")
def set_camera(index: int):
    ok = CAM.open(index)
    return JSONResponse({"success": bool(ok), "camera_index": CAM.get_index(), "camera_name": CAM.get_name(CAM.get_index())})

@app.post("/cameras/probe")
def cameras_probe():
    new_list = CAM.reprobe()
    return JSONResponse({"cameras":[{"index":i,"name":CAM.get_name(i)} for i in new_list], "current": CAM.get_index()})

# session endpoints
@app.post("/class/start")
def api_start_class(payload: dict = None):
    try:
        start_iso = None
        duration = CLASS_DURATION_MIN
        interval_min = INTERVAL_MIN
        if payload:
            start_iso = payload.get("start_iso")
            duration = int(payload.get("duration_min", duration))
            interval_min = int(payload.get("interval_min", interval_min))
        sid = create_session(start_time=start_iso, duration_min=duration, interval_min=interval_min)
        return JSONResponse({"status":"ok","session_id":sid})
    except Exception as e:
        return JSONResponse({"status":"error","message":str(e)}, status_code=500)

@app.post("/class/stop")
def api_stop_class(payload: dict = None):
    try:
        sid = None
        if payload and payload.get("session_id"):
            sid = payload.get("session_id")
        else:
            sid, _ = get_active_session()
        if not sid:
            return JSONResponse({"status":"error","message":"no active session"}, status_code=404)
        stop_session(sid)
        return JSONResponse({"status":"ok","session_id":sid})
    except Exception as e:
        return JSONResponse({"status":"error","message":str(e)}, status_code=500)

@app.get("/class/current")
def api_current_class():
    sid, sess = get_active_session()
    if not sid:
        return JSONResponse({"active": False})
    return JSONResponse({"active": True, "session_id": sid, "start_iso": sess["start_iso"], "intervals": sess["intervals"], "interval_min": sess["interval_min"]})

def get_session_summary(session_id):
    sess = SESSIONS.get(session_id)
    if not sess:
        return None
    intervals = sess.get("intervals", max(1, sess.get("duration_min",CLASS_DURATION_MIN)//sess.get("interval_min",INTERVAL_MIN)))
    attendance = {}
    attendance.update(sess.get("attendance", {}))
    try:
        with open(INTERVAL_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] == session_id:
                    name = row[1]
                    idx = int(row[2])
                    attendance.setdefault(name, [])
                    if idx not in attendance[name]:
                        attendance[name].append(idx)
    except Exception:
        pass
    threshold = max(1, int((intervals * 0.6) + 0.9999))
    all_students = sorted({*attendance.keys(), *([n for n in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH,n))])})
    summary = {}
    for name in all_students:
        present_intervals = sorted(set(attendance.get(name, [])))
        vect = [1 if i in present_intervals else 0 for i in range(intervals)]
        count = sum(vect)
        present_flag = (count >= threshold)
        summary[name] = {"intervals": vect, "present_count": count, "present": present_flag}
    return {
        "session_id": session_id,
        "start_iso": sess["start_iso"],
        "duration_min": sess["duration_min"],
        "interval_min": sess["interval_min"],
        "intervals": intervals,
        "threshold": threshold,
        "summary": summary
    }

@app.get("/class/{session_id}/summary")
def api_summary(session_id: str):
    summary = get_session_summary(session_id)
    if not summary:
        return JSONResponse({"status":"error","message":"session not found"}, status_code=404)
    return JSONResponse(summary)

@app.post("/reload")
def reload_encodings():
    load_dataset()
    return {"status":"ok","known_faces": len(student_names)}

def run_dashboard():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# -------------------------
# Face recognition main loop
# -------------------------
def face_recognition_loop():
    print("Starting Attendance System (webcam). Press 'r' to register, 'q' to quit.")
    while True:
        ret, frame = CAM.read()
        if not ret:
            time.sleep(0.05)
            continue
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_small, model=ENCODING_MODEL)
        encs = face_recognition.face_encodings(rgb_small, boxes)
        unknown_present = False
        for enc, box in zip(encs, boxes):
            if len(student_encodings) == 0:
                name = "Unknown"
                unknown_present = True
            else:
                matches = face_recognition.compare_faces(student_encodings, enc, tolerance=0.5)
                distances = face_recognition.face_distance(student_encodings, enc)
                name = "Unknown"
                if True in matches:
                    best_idx = np.argmin(distances)
                    name = student_names[best_idx]
                    # **NEW:** record interval presence; final attendance will be appended when threshold met
                    add_interval_presence(name, datetime.utcnow())
                else:
                    unknown_present = True
            top, right, bottom, left = [v * 4 for v in box]
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cam_idx = CAM.get_index()
        cam_name = CAM.get_name(cam_idx)
        cv2.putText(frame, f"{cam_name} ({cam_idx})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Attendance System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if unknown_present:
                print("[INFO] Registration triggered. Enter name in terminal.")
                register_student_interactive()
            else:
                print("[INFO] No unknown face detected.")
        elif key == ord("n"):
            av = CAM.get_available()
            if av:
                try:
                    cur = CAM.get_index()
                    pos = av.index(cur) if cur in av else -1
                    new = av[(pos+1) % len(av)] if av else cur
                    CAM.open(new)
                except Exception:
                    pass
        elif key == ord("b"):
            av = CAM.get_available()
            if av:
                try:
                    cur = CAM.get_index()
                    pos = av.index(cur) if cur in av else 0
                    new = av[(pos-1) % len(av)] if av else cur
                    CAM.open(new)
                except Exception:
                    pass
    CAM.release()
    cv2.destroyAllWindows()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    SESSIONS.update(load_sessions())
    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()
    face_recognition_loop()
    print("[INFO] Exiting program.")
