import cv2
import mediapipe
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from flask import Flask, request

app = Flask(__name__)

def prediction(video_file, lstm_model):
    actions = ['banana', 'bar', 'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom']

    sequence = []
    frame_num = 0
    while True:
        ret, frame = video_file.read()
        if not ret:
            break

        # rotate video right way up
        (h, w) = frame.shape[:2]
        rotpoint = (w // 2, h // 2)
        rotmat = cv2.getRotationMatrix2D(rotpoint, 180, 1.0)
        dim = (w, h)
        intermediateFrame = cv2.warpAffine(frame, rotmat, dim)

        # cropping
        size = intermediateFrame.shape
        finalFrame = intermediateFrame[80:(size[0] - 200), 30:(size[1] - 30)]

        # keypoint prediction
        image = cv2.cvtColor(finalFrame, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = holistic.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR

        # extract and append keypoints
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        keypoints = np.concatenate([pose, lh, rh])
        sequence.append(keypoints)

        if len(sequence) == 50:
            break

    sequence = np.expand_dims(sequence, axis=0)[0]

    res = lstm_model.predict(np.expand_dims(sequence, axis=0))
    return actions[np.argmax(res)]

lstm_model = load_model('C:/Users/zoro/Documents/signchatter/NewAction3.h5')
mp_holistic = mediapipe.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    try:
        video_file = cv2.VideoCapture(file)
        result = prediction(video_file, lstm_model)
        return result, 200
if __name__ == '__main__':
    app.run()
