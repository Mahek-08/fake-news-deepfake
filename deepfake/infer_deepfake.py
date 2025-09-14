import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = 'deepfake/models/deepfake_xception.h5'
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)


def frame_predict(frame):
    # frame: BGR image from cv2
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, 0)
    pred = model.predict(img)[0][0]
    # pred ~ probability of class 1 (fake)
    return float(pred)


def video_predict(video_path, sample_rate=15, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    scores = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_rate == 0:
            score = frame_predict(frame)
            scores.append(score)
        i += 1
    cap.release()
    if not scores:
        return {'score': 0.0, 'label': 'unknown'}
    avg = float(np.mean(scores))
    label = 'fake' if avg >= threshold else 'real'
    return {'score': avg, 'label': label}

if __name__ == '__main__':
    print(video_predict('some_video.mp4'))
