import os 

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# face detection model
PROTOTX_PATH = os.path.join(MODELS_DIR, 'deploy.prototxt.txt')
MODEL_PATH = os.path.join(MODELS_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

# landmark path
LANDMARKS_PATH = os.path.join(MODELS_DIR, 'shape_predictor_68_face_landmarks.dat')

YAW_THRESHOLD = 25 # batas toleransi menoleh kiri/kanan
PITCH_THRESHOLD = 25 # batas toleransi menoleh atas/bawah

FACE_CONFIDENCE_THRESHOLD = 0.5 

WEBCAM_ID = 0
