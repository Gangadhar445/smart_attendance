import os
import time
import json
import csv
import math
import threading
from datetime import datetime

import cv2
import face_recognition
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Config / Paths

DATASET_PATH = "dataset"
ATTENDANCE_FILE = "attendance.csv"
INTERVAL_FILE = "interval_attendance.csv"
TEMPLATES_DIR = "templates"
ENCODING_MODEL = "hog"   # 'hog' or 'cnn'
CAMERA_NAMES_FILE = "camera_names.json"
PROBE_MAX = 6
SESSIONS_FILE = "sessions.json"
# in-memory guard to avoid writing the same final attendance repeatedly
_MARKED_TODAY_LOCK = threading.Lock()
_MARKED_TODAY_DATE = None   # string YYYY-MM-DD in UTC
_MARKED_TODAY_SET = set()


# Ensure files exist
os.makedirs(DATASET_PATH, exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

if not os.path.exists(INTERVAL_FILE):
    with open(INTERVAL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "name", "interval_index", "timestamp_utc"])


# Sessions store

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

def create_session(duration_min):
    """Create session with fixed 6 intervals and interval_min = duration_min / 6."""
    with _sessions_lock:
        sid = str(int(time.time() * 1000))
        start_iso = datetime.utcnow().isoformat()
        intervals = 6
        interval_min = float(duration_min) / float(intervals)
        SESSIONS[sid] = {
            "start_iso": start_iso,
            "duration_min": float(duration_min),
            "interval_min": float(interval_min),
            "intervals": intervals,
            "active": True,
            "attendance": {}   # name -> list of interval indices (in-memory)
        }
        save_sessions(SESSIONS)
        print(f"[SESSIONS] created {sid} start={start_iso} duration={duration_min}m interval_min={interval_min}m (6 intervals)")
        return sid

def stop_session(session_id):
    with _sessions_lock:
        sess = SESSIONS.get(session_id)
        if sess:
            sess["active"] = False
            sess["attendance"] = {}   # clear in-memory attendance for that session
            SESSIONS[session_id] = sess
            save_sessions(SESSIONS)
            print(f"[SESSIONS] stopped {session_id}")
            return True
    return False

def get_active_session():
    with _sessions_lock:
        for sid, sess in SESSIONS.items():
            if sess.get("active"):
                return sid, sess
    return None, None

def compute_interval_index_for_session(sess, dt_utc):
    start = datetime.fromisoformat(sess["start_iso"])
    delta = dt_utc - start
    if delta.total_seconds() < 0:
        return None
    dur_seconds = sess["duration_min"] * 60.0
    if delta.total_seconds() >= dur_seconds:
        return None
    interval_seconds = sess["interval_min"] * 60.0
    idx = int(delta.total_seconds() // interval_seconds)
    if idx < 0 or idx >= sess["intervals"]:
        return None
    return idx

# Interval CSV utilities

def interval_csv_has(session_id, name, interval_index):
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

def clear_interval_csv():
    try:
        with open(INTERVAL_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "name", "interval_index", "timestamp_utc"])
        print("[INTERVAL CSV] cleared (header only).")
    except Exception as e:
        print("[INTERVAL CSV] clear failed:", e)


# Attendance helpers

def already_marked_today(name):
    """Return True if ATTENDANCE_FILE already has name for today's UTC date OR in-memory set."""
    try:
        # in-memory fast check first
        utc_today = datetime.utcnow().strftime("%Y-%m-%d")
        with _MARKED_TODAY_LOCK:
            global _MARKED_TODAY_DATE, _MARKED_TODAY_SET
            # refresh in-memory set if date changed
            if _MARKED_TODAY_DATE != utc_today:
                _MARKED_TODAY_DATE = utc_today
                _MARKED_TODAY_SET = set()
                # load today's names from CSV into memory set to avoid duplicates
                try:
                    with open(ATTENDANCE_FILE, "r", newline="") as f:
                        reader = csv.reader(f)
                        next(reader, None)
                        for row in reader:
                            if len(row) >= 2 and row[1] == utc_today:
                                _MARKED_TODAY_SET.add(row[0])
                except FileNotFoundError:
                    pass
                except Exception:
                    # fall back to scanning file later
                    pass

            if name in _MARKED_TODAY_SET:
                return True

        # If not found in-memory, do a file scan as a fallback (robust)
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == utc_today:
                    # also add to in-memory set for future checks
                    with _MARKED_TODAY_LOCK:
                        _MARKED_TODAY_SET.add(name)
                    return True
    except Exception:
        # If anything goes wrong, be conservative and return False so we can still try to write.
        pass
    return False


def append_attendance_now(name, when_local=None):
    """
    Append a row to ATTENDANCE_FILE for name with the given local timestamp.
    Uses in-memory guard to ensure we append at most once per student per UTC day.
    """
    try:
        utc_today = datetime.utcnow().strftime("%Y-%m-%d")
        with _MARKED_TODAY_LOCK:
            global _MARKED_TODAY_DATE, _MARKED_TODAY_SET
            # refresh date and set if needed
            if _MARKED_TODAY_DATE != utc_today:
                _MARKED_TODAY_DATE = utc_today
                _MARKED_TODAY_SET = set()
                # load existing today's rows into set
                try:
                    with open(ATTENDANCE_FILE, "r", newline="") as f:
                        reader = csv.reader(f)
                        next(reader, None)
                        for row in reader:
                            if len(row) >= 2 and row[1] == utc_today:
                                _MARKED_TODAY_SET.add(row[0])
                except FileNotFoundError:
                    pass
                except Exception:
                    pass

            # if in-memory shows already present -> skip
            if name in _MARKED_TODAY_SET:
                return False

            # Double-check file as last safety (read again)
            try:
                with open(ATTENDANCE_FILE, "r", newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2 and row[0] == name and row[1] == utc_today:
                            _MARKED_TODAY_SET.add(name)
                            return False
            except FileNotFoundError:
                pass
            except Exception:
                pass

            # OK: perform append
            now_local = when_local or datetime.now()
            date_str_local = now_local.strftime("%Y-%m-%d")
            time_str_local = now_local.strftime("%H:%M:%S")
            with open(ATTENDANCE_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, date_str_local, time_str_local])

            # update in-memory set
            _MARKED_TODAY_SET.add(name)
            print(f"[FINAL MARK] {name} on {date_str_local} at {time_str_local}")
            return True

    except Exception as e:
        print("[ATTENDANCE CSV] write failed:", e)
        return False


# Add interval presence & threshold check (65% rule)

def add_interval_presence(name, timestamp_utc):
    sid, sess = get_active_session()
    if not sid:
        return None
    idx = compute_interval_index_for_session(sess, timestamp_utc)
    if idx is None:
        return None
    # write interval hit if not duplicate
    if not interval_csv_has(sid, name, idx):
        append_interval_csv(sid, name, idx, timestamp_utc)
    # update in-memory
    with _sessions_lock:
        attendance = sess.setdefault("attendance", {})
        arr = attendance.setdefault(name, [])
        if idx not in arr:
            arr.append(idx)
        SESSIONS[sid] = sess
        save_sessions(SESSIONS)
    # compute total unique intervals for this student
    indices_set = set(sess.get("attendance", {}).get(name, []))
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
    intervals_total = sess.get("intervals", 6)
    threshold = math.ceil(0.65 * intervals_total)  # 65% rule
    if count >= threshold:
        if not already_marked_today(name):
            append_attendance_now(name)
    return idx


# Dataset load (encodings)

student_encodings = []
student_names = []

def load_dataset():
    global student_encodings, student_names
    student_encodings = []
    student_names = []
    if not os.path.exists(DATASET_PATH):
        print("[WARN] no dataset dir")
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
                print(f"[WARN] failed {img_path}: {e}")
    print(f"[INFO] loaded {len(student_names)} known face images.")

load_dataset()


# CameraManager (camera closed by default)

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
        print(f"[CAM] probed: {found}")
        self.available = found
        return found

    def load_camera_names(self):
        if os.path.exists(CAMERA_NAMES_FILE):
            try:
                with open(CAMERA_NAMES_FILE, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                pass
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
                self.capture = None
                return False
            self.capture = cap
            self.index = index
            if index not in self.available:
                self.available.append(index)
            print(f"[CAM] opened camera {index}")
            return True

    def read(self):
        with self.lock:
            if self.capture is None:
                return False, None
            try:
                ret, frame = self.capture.read()
                return ret, frame
            except Exception:
                return False, None

    def release(self):
        with self.lock:
            try:
                if self.capture is not None:
                    self.capture.release()
            except Exception:
                pass
            self.capture = None
            print("[CAM] released camera")

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
                if ok:
                    new_found.append(i)
            self.available = new_found
            for idx in self.available:
                if str(idx) not in self.names:
                    self.names[str(idx)] = f"Camera {idx}"
            if self.index not in self.available:
                if self.available:
                    try:
                        self.open(self.available[0])
                    except Exception as e:
                        print("[CAM] reprobe open failed:", e)
                else:
                    self.release()
            self.save_camera_names()
            print(f"[CAM] reprobe result: {self.available}")
            return list(self.available)

CAM = CameraManager(start_index=0, probe_max=PROBE_MAX)


# Registration (auto-capture)

def normalize_name(s: str) -> str:
    return "".join(s.lower().split())

def get_existing_names():
    if not os.path.exists(DATASET_PATH):
        return []
    return [n for n in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, n))]

def register_student_interactive():
    camera_was_closed = CAM.capture is None
    if camera_was_closed:
        # try to open first available camera for registration
        cams = CAM.probe_cameras()
        if cams:
            CAM.open(cams[0])
    existing_names = get_existing_names()
    existing_norm = {normalize_name(n): n for n in existing_names}
    while True:
        name = input("Enter student name: ").strip()
        if not name:
            print("[INFO] Name empty. Aborting registration.")
            if camera_was_closed:
                CAM.release()
            return
        if normalize_name(name) in existing_norm:
            print(f"[WARN] A similar/exact name already exists: '{existing_norm[normalize_name(name)]}'")
            choice = input("Enter 'r' to retry or 'a' to abort: ").strip().lower()
            if choice == "r":
                continue
            else:
                print("[INFO] Registration aborted.")
                if camera_was_closed:
                    CAM.release()
                return
        # fuzzy check omitted for brevity if not desired
        break

    person_dir = os.path.join(DATASET_PATH, name)
    os.makedirs(person_dir, exist_ok=True)
    max_shots = 5
    delay_between = 0.6
    capturing = False
    captured = 0
    print("[INFO] Registration preview started using camera index", CAM.get_index())
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
        if camera_was_closed:
            CAM.release()

    if os.path.exists(person_dir) and len(os.listdir(person_dir)) > 0:
        load_dataset()
        print(f"[INFO] Registered '{name}' with {len(os.listdir(person_dir))} images.")
    else:
        print("[INFO] No images saved. Registration aborted or incomplete.")


# FastAPI

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
    # total students = number of folders in DATASET_PATH (registered students)
    try:
        total_students = len([n for n in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, n))])
    except Exception:
        total_students = 0

    # Load attendance CSV
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        # ensure Date column present and parsed
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")
        else:
            df["Date_str"] = ""
    except Exception:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Date_str"])

    # today's date in LOCAL (use same local format you write in ATTENDANCE_FILE)
    today_local_str = datetime.now().strftime("%Y-%m-%d")

    # build today present list robustly
    if not df.empty:
        today_present_students = df[df["Date_str"] == today_local_str]["Name"].unique().tolist()
    else:
        today_present_students = []

    today_attendance = len(today_present_students)
    today_absent = max(total_students - today_attendance, 0)

    # Attendance % per student across recorded days (use unique dates from CSV)
    attendance_percentage = {}
    if not df.empty and len(df["Date_str"].unique()) > 0:
        unique_days = len(df["Date_str"].unique())
        for student in sorted(list(set(df["Name"].unique()).union(set([n for n in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, n))])))):
            attendance_percentage[student] = (len(df[df["Name"] == student]) / unique_days) * 100
    else:
        # no data yet -> all zeros (but still list registered students)
        for student in [n for n in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, n))]:
            attendance_percentage[student] = 0.0

    # Engagement (counts)
    if not df.empty:
        engagement = df["Name"].value_counts().reset_index()
        engagement.columns = ["Name", "Attendance Count"]
        engage_json = {
            "students": engagement["Name"].tolist(),
            "counts": engagement["Attendance Count"].tolist(),
            "total_days": len(df["Date_str"].unique())
        }
    else:
        engage_json = {"students": [], "counts": [], "total_days": 0}

    # Heatmap (dates from df)
    heatmap_students = sorted(list(set(df["Name"].unique()).union(set([n for n in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, n))]))))
    heatmap_dates = sorted(df["Date_str"].dropna().unique().tolist())
    heatmap_matrix = []
    for student in heatmap_students:
        row = []
        for d in heatmap_dates:
            present = not df[(df["Name"] == student) & (df["Date_str"] == d)].empty
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

# Cameras endpoints
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

# Session endpoints
@app.post("/class/start")
def api_start_class(payload: dict = None):
    try:
        duration = 50.0
        if payload and payload.get("duration_min"):
            duration = float(payload.get("duration_min"))
        # reprobe cameras first so we know what's available
        CAM.reprobe()
        # open first available camera if any — else return warning but still create session
        cams = CAM.get_available()
        if cams:
            CAM.open(cams[0])
        sid = create_session(duration)
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
        # clear interval CSV (header only)
        clear_interval_csv()
        # release camera
        CAM.release()
        return JSONResponse({"status":"ok","session_id":sid})
    except Exception as e:
        return JSONResponse({"status":"error","message":str(e)}, status_code=500)

@app.get("/class/current")
def api_current_class():
    sid, sess = get_active_session()
    if not sid:
        return JSONResponse({"active": False})
    return JSONResponse({"active": True, "session_id": sid, "start_iso": sess["start_iso"], "intervals": sess["intervals"], "interval_min": sess["interval_min"], "duration_min": sess["duration_min"]})

def get_session_summary(session_id):
    sess = SESSIONS.get(session_id)
    if not sess:
        return None
    intervals = sess.get("intervals", 6)
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
    threshold = math.ceil(0.65 * intervals)
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


# Face recognition loop (background)

def face_recognition_loop():
    print("Face recognition thread started. Camera will be opened only when session starts.")
    while True:
        # only run when camera opened and a session is active
        if CAM.capture is None:
            time.sleep(0.15)
            continue
        sid, sess = get_active_session()
        if not sid:
            # no active session, don't process frames (but camera may be open for registration)
            time.sleep(0.15)
            continue
        ret, frame = CAM.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_small, model=ENCODING_MODEL)
        encs = face_recognition.face_encodings(rgb_small, boxes)
        for enc, box in zip(encs, boxes):
            if len(student_encodings) == 0:
                name = "Unknown"
            else:
                matches = face_recognition.compare_faces(student_encodings, enc, tolerance=0.5)
                distances = face_recognition.face_distance(student_encodings, enc)
                name = "Unknown"
                if True in matches:
                    best_idx = int(np.argmin(distances))
                    name = student_names[best_idx]
                    # record interval presence (INTERVAL_FILE) and check threshold (may append final attendance)
                    add_interval_presence(name, datetime.utcnow())
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
            CAM.release()
            cv2.destroyAllWindows()
            break
        elif key == ord("r"):
            register_student_interactive()
    print("[FACE LOOP] exiting")


# Main

if __name__ == "__main__":
    # restore any saved sessions
    SESSIONS.update(load_sessions())

    # Start FastAPI dashboard in a daemon thread so it doesn't block main thread
    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()

    # IMPORTANT: run face_recognition_loop() in the MAIN thread (not a daemon/background thread)
    # GUI functions (cv2.imshow) work reliably only on the main thread on many platforms.
    try:
        face_recognition_loop()
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt, shutting down...")
    finally:
        # ensure camera released and OpenCV windows closed
        CAM.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[MAIN] Exiting.")
