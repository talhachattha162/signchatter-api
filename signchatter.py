import numpy as np
import mediapipe
import cv2
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model


def prediction(camera_index, lstm_model):
    actions = ['banana', 'bar', 'basement', 'basketball', 'bath', 'bathroom', 'bear', 'beard', 'bed', 'bedroom']
    sequence = []
    
    cap = cv2.VideoCapture(camera_index)  # Open the camera of the mobile device

    mp_holistic = mediapipe.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

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
        print(actions[np.argmax(res)])

    cap.release()
    cv2.destroyAllWindows()

lstm_model = load_model('NewAction3.h5')
prediction(0, lstm_model)  # Pass 0 as camera_index to use the default camera
