import cv2
# import torchvision.transforms as transforms
import numpy as np
# from model import ASL_model
# from PIL import Image
import pandas as pd
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import sys

# MODEL_PATH = "D:/2025/ASL_model/Codev2/model.h5"
# LABELS_PATH = "D:/2025/ASL_model/Codev2/labels.csv"
if getattr(sys, 'frozen', False):
    # Đang chạy từ tệp .exe
    BASE_DIR = sys._MEIPASS
else:
    # Đang chạy từ mã Python
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")

labels = pd.read_csv(LABELS_PATH)
labels = labels['label'].tolist()
print(labels)

# Tải mô hình
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

def mediapipe_detection(image, holistic_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = holistic_model.process(image)        # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

# def extract_keypoints(results):
#     # Lấy keypoints của pose (33 điểm)
#     pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)

#     # Lấy keypoints của bàn tay trái (21 điểm)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

#     # Lấy keypoints của bàn tay phải (21 điểm)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

#     # Kết hợp tất cả keypoints
#     return np.concatenate([pose, lh, rh])

def extract_keypoints(results):
    """
    Trích xuất keypoints từ Mediapipe Holistic:
    - Pose: 33 keypoints
    - Left hand: 21 keypoints
    - Right hand: 21 keypoints
    Tổng cộng: 75 keypoints.

    Args:
        results: Kết quả từ Mediapipe Holistic.

    Returns:
        np.ndarray: Mảng keypoints có kích thước (75,).
    """
    if results.pose_landmarks:
        # Lấy tọa độ hai vai
        left_shoulder = results.pose_landmarks.landmark[11]  # Left shoulder
        right_shoulder = results.pose_landmarks.landmark[12]  # Right shoulder
        # Tính trung điểm
        ref_x = (left_shoulder.x + right_shoulder.x) / 2
        ref_y = (left_shoulder.y + right_shoulder.y) / 2
        ref_z = (left_shoulder.z + right_shoulder.z) / 2
        # Chuẩn hóa pose landmarks
        pose = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z] 
                        for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 3)
        ref_x, ref_y, ref_z = 0, 0, 0

    # Chuẩn hóa tay trái
    if results.left_hand_landmarks and results.pose_landmarks:
        lh = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z] 
                      for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # Chuẩn hóa tay phải
    if results.right_hand_landmarks and results.pose_landmarks:
        rh = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z] 
                      for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])

def Z_score_normalization(data):
    """
    Chuẩn hóa dữ liệu bằng phương pháp Z-score.

    Args:
        data (np.ndarray): Dữ liệu đầu vào.

    Returns:
        np.ndarray: Dữ liệu đã được chuẩn hóa.
    """
    MEAN_PATH = os.path.join(BASE_DIR, "mean.npy")
    STD_PATH = os.path.join(BASE_DIR, "std.npy")

    # Load dữ liệu
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    return (data - mean) / std

holistic_model = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = (fps + 29) // 30
    count = 0
    keypoints = []
    pre_label = None
    confidence = 0.0
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        if count % interval == 0:
            cop_frame = frame.copy()
            cop_frame = cv2.resize(cop_frame, (224, 224))
            _, results = mediapipe_detection(cop_frame, holistic_model)
            keypoint = extract_keypoints(results)
            keypoints.append(keypoint)
            keypoints = keypoints[-30:]
            if len(keypoints) == 30:

                keypoints_np = np.array(keypoints)
                keypoints_np = Z_score_normalization(keypoints_np)
                keypoints_np = keypoints_np.reshape(30, 225) 
                outputs = model.predict(np.array([keypoints_np]), verbose=0)
                predicted = np.argmax(outputs, axis=1)
                confidence = float(np.max(outputs))
                temp = labels[predicted[0]]
                print(f"Dự đoán: {temp} (Độ tự tin: {confidence:.2f}%)")
                if confidence > 0.75:
                    pre_label = temp
                    
        count += 1
        count = count % interval
        if pre_label:
            cv2.putText(frame, f"{pre_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# pyinstaller --onefile --add-data "model.h5;." --add-data "labels.csv;." --add-data "mean.npy;." --add-data "std.npy;." --add-data "C:\Users\ASUS\AppData\Local\Programs\Python\Python310\lib\site-packages\mediapipe\modules\holistic_landmark;mediapipe\modules\holistic_landmark" test_model.py







