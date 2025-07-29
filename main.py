import cv2
import imutils
import time
from controller.config import WEBCAM_ID
from controller.pose_estimator import PoseEst
from controller.tracker_logic import WorktTimeTracker

def main():
    pose_estimator = PoseEst()
    work_tracker = WorktTimeTracker()

    vs = cv2.VideoCapture(WEBCAM_ID)
    if not vs.isOpened():
        print(f"Error: Could not open webcam ID {WEBCAM_ID}.")
        return
    
    time.sleep(2.0)

    while True:
        ret, frame = vs.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame = imutils.resize(frame, width=800)

        face_detected, pitch, yaw, roll, face_rect_coords = pose_estimator.process_frame(frame)

        is_facing_webcam = False 
        bbox_color = (0, 0, 255)
        status_text = "Not Facing Webcam"

        if face_detected and pitch is not None and yaw is not None:
            if face_rect_coords:
                (startX, startY, endX, endY) = face_rect_coords

                status_text = 'Away From Keyboard'

                # print(f"[DEBUG] Yaw: {yaw}, Pitch: {pitch}, Is Facing: {pose_estimator.is_facing_forward(pitch, yaw)}")
                if pose_estimator.is_facing_forward(pitch, yaw):
                    is_facing_webcam = True
                    bbox_color = (0, 255, 0)
                    status_text = 'Work'

                cv2.rectangle(frame, (startX, startY), (endX, endY), bbox_color, 2)
                cv2.putText(frame, status_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)

                # Konversi menit ke format hours, minutes, dan seconds dengan dua digit
                total_minutes = work_tracker.get_total_work_time_minutes()
                hours = int(total_minutes // 60)
                minutes = int(total_minutes % 60)
                seconds = int((total_minutes * 60) % 60)  # Konversi desimal menit ke detik
                time_text = f"{hours:01d}h, {minutes:02d}min, {seconds:02d}sec"

                # Tampilkan waktu kerja
                cv2.putText(frame, f'Time Work: {time_text}', (endX - 200, endY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)

        work_tracker.update(is_facing_webcam)


        cv2.imshow('Work Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()
    pose_estimator.close()
    # Tampilkan total waktu kerja akhir
    total_minutes_final = work_tracker.get_total_work_time_minutes()
    hours_final = int(total_minutes_final // 60)
    minutes_final = int(total_minutes_final % 60)
    seconds_final = int((total_minutes_final * 60) % 60)  
    time_text_final = f"{hours_final:02d} hours, {minutes_final:02d} minutes, {seconds_final:02d} seconds"
    print(f'Overall Time Work: {time_text_final}')

if __name__ == '__main__':
    main()