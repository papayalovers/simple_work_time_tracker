import time 

class WorktTimeTracker:
    def __init__(self):
        self.total_work_time = 0 # total waktu kerja dalam detik
        self.session_start_time = None # waktu mulai sesi kerja
        self.last_frame_time = time.time() # waktu frame terakhir

    def update(self, is_facing_webcam):
        current_time = time.time()
        delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if is_facing_webcam:
            if self.session_start_time is None:
                self.session_start_time = current_time
            self.total_work_time += delta_time
        else:
            if self.session_start_time is not None:
                self.session_start_time = None

    def get_total_work_time_seconds(self):
        return self.total_work_time 

    def get_total_work_time_minutes(self):
        return self.total_work_time / 60