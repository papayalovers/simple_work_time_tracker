# Work Time Tracker
A simple app to track work time using face pose detection with MediaPipe and OpenCV.

## Setup
Create new environtment:
`python3.9 -m venv mediapipe_env`

Activate the environtment:
`source mediapipe_env/bin/activate`

Install libraries:
`pip install -r requirements.txt`

Run the App:
`python main.py`

## Project Structure
```
work_time_tracker/
│
├── controller/
│   ├── config.py
│   ├── pose_estimator.py
│   └── tracker_logic.py
│
├── main.py
├── requirements.txt
├── mediapipe_env/  (created after setup)
└── README.md
```

## Notes
- Press `q` to quit the app.
- Edit `controller/config.py` to adjust settings like webcam ID or thresholds.