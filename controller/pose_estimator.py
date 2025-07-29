# pose_estimator.py

import cv2
import mediapipe as mp
import numpy as np
from controller.config import YAW_THRESHOLD, PITCH_THRESHOLD 

class PoseEst:
    def __init__(self):
        print("[INFO] Memuat MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 3D model points (objek referensi wajah 3D) dan indeks landmark MediaPipe yang sesuai.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
            (0.0, -330.0, -65.0),        # Chin (landmark 152)
            (-225.0, 170.0, -135.0),     # Left eye outer corner (landmark 33)
            (225.0, 170.0, -135.0),      # Right eye outer corner (landmark 263)
            (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
            (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
        ], dtype="double")

        self.mediapipe_indices = [1, 152, 33, 263, 61, 291]
        
        print(f"[DEBUG] PoseEstimator diinisialisasi dengan YAW_THRESHOLD={YAW_THRESHOLD}, PITCH_THRESHOLD={PITCH_THRESHOLD}")


    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        face_detected = False
        pitch, yaw, roll = None, None, None
        face_rect = None 

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                # Dapatkan dimensi frame
                (h, w) = frame.shape[:2]

                # Konversi landmark ke koordinat gambar (piksel)
                image_points = []
                for idx in self.mediapipe_indices:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    image_points.append((x, y))

                image_points = np.array(image_points, dtype='double')

                # Estimasi parameter kamera
                focal_length = 600  # Nilai tetap, sesuaikan jika perlu
                center = (w / 2, h / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype='double')

                dist_coeffs = np.zeros((4, 1))  # Koefisien distorsi lensa (asumsi nol)

                # Estimasi pose menggunakan solvePnP
                (success, rotation_vector, translation_vector) = cv2.solvePnP(
                    self.model_points, image_points, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Konversi rotation vector ke Euler angles
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    
                    # Ekstraksi sudut Euler dari rotation_matrix
                    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
                    singular = sy < 1e-6

                    if not singular:
                        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # pitch
                        y = np.arctan2(-rotation_matrix[2, 0], sy)                    # yaw
                        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])   # roll
                    else:
                        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])  # pitch
                        y = np.arctan2(-rotation_matrix[2, 0], sy)                    # yaw
                        z = 0

                    # Konversi ke derajat dan normalisasi
                    pitch = np.degrees(x)
                    pitch = ((pitch + 180) % 360) - 180
                    if pitch > 90:
                        pitch = pitch - 180
                    elif pitch < -90:
                        pitch = pitch + 180

                    yaw = np.degrees(y)
                    yaw = ((yaw + 180) % 360) - 180
                    roll = np.degrees(z)
                    roll = ((roll + 180) % 360) - 180

                # Mendapatkan bounding box dari landmarks
                x_coords = [int(l.x * w) for l in face_landmarks.landmark]
                y_coords = [int(l.y * h) for l in face_landmarks.landmark]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                face_rect = (max(0, min_x), max(0, min_y), min(w, max_x), min(h, max_y))
                
                break

        return face_detected, pitch, yaw, roll, face_rect
    
    def is_facing_forward(self, pitch, yaw):
        # Pengecekan nilai None sebelum operasi abs
        if pitch is None or yaw is None:
            return False
        return abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()