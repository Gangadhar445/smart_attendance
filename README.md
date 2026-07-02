# Smart Attendance System

## Overview

Smart Attendance System is an AI-powered attendance management application that automatically detects and recognizes student faces using a camera feed and marks attendance in real time. The system provides a web-based dashboard for monitoring attendance records and analytics.

## Features

* Automatic face detection and recognition
* Real-time attendance marking
* Duplicate attendance prevention
* Live camera feed integration
* Attendance analytics dashboard
* Daily attendance tracking
* Attendance percentage calculation
* Interactive charts and visualizations
* FastAPI-based web application

## Technologies Used

* Python
* FastAPI
* OpenCV
* face_recognition
* NumPy
* Pandas
* Plotly
* Jinja2
* Uvicorn

## Project Structure

```text
SmartAttendance/
│
├── dataset/
│   └── Student Images
│
├── templates/
│   └── dashboard_realtime.html
│
├── static/
│
├── attendance.csv
├── final.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SmartAttendance
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

#### macOS/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Create a `dataset` folder.
2. Add student images.
3. Name images using student identifiers or names according to the application's format.

Example:

```text
dataset/
├── John.jpg
├── Alice.jpg
├── David.jpg
```

## Running the Application

Start the application using:

```bash
python final.py
```

Or if using Uvicorn directly:

```bash
uvicorn final:app --reload
```

## Access the Dashboard

Open your browser and visit:

```text
http://127.0.0.1:8000
```

## Attendance Output

Attendance records are stored in:

```text
attendance.csv
```

Each record contains:

* Student Name
* Date
* Time

## Dashboard Features

* Real-time attendance table
* Attendance statistics
* Attendance percentage tracking
* Weekly attendance trends
* Student engagement insights
* Interactive visualizations

## Future Enhancements

* Database integration
* Cloud deployment
* Multi-camera support
* Face mask detection
* Student registration portal
* Email notifications
* Attendance reports in PDF format

## License

This project is developed for educational, research, and demonstration purposes.
