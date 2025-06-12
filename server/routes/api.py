from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pandas as pd
import base64
import os
import time
import requests
from collections import deque

api = Blueprint('api', __name__)

# Đường dẫn đến model và các file liên quan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "../models/labels.csv")
MEAN_PATH = os.path.join(BASE_DIR, "../models/mean.npy")
STD_PATH = os.path.join(BASE_DIR, "../models/std.npy")

# Cấu hình web server
WEB_API_URL = "http://127.0.0.1:5002/api"

# Tải model và các file liên quan
model = load_model(MODEL_PATH)
labels = pd.read_csv(LABELS_PATH)['label'].tolist()
mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)

# Tạo buffer cho frame
frame_buffer = deque(maxlen=5)  # Lưu trữ 5 frame gần nhất

# Tải Mediapipe Holistic model với cấu hình tối ưu
holistic_model = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
    enable_segmentation=False,
    static_image_mode=False
)

def send_to_web(result_type, data):
    """Gửi dữ liệu lên web server"""
    try:
        if result_type == 'result':
            response = requests.post(f"{WEB_API_URL}/update-result", json=data, timeout=1)
            response.raise_for_status()
        elif result_type == 'stream':
            response = requests.post(f"{WEB_API_URL}/update-stream", json=data, timeout=1)
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending to web server: {e}")

def mediapipe_detection(image, holistic_model):
    """
    Sử dụng Mediapipe để phát hiện keypoints.
    """
    try:
        # Chuyển đổi ảnh sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Tăng hiệu suất xử lý
        
        # Thêm frame vào buffer
        frame_buffer.append(image)
        
        # Xử lý frame mới nhất
        results = holistic_model.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        print(f"Error in Mediapipe detection: {e}")
        # Thử xử lý frame từ buffer nếu có lỗi
        if len(frame_buffer) > 1:
            try:
                prev_frame = frame_buffer[-2]  # Lấy frame trước đó
                results = holistic_model.process(prev_frame)
                return image, results
            except Exception as retry_error:
                print(f"Error in retry detection: {retry_error}")
        return image, None

def extract_keypoints(results):
    """
    Trích xuất keypoints từ Mediapipe Holistic.
    """
    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]
        ref_x = (left_shoulder.x + right_shoulder.x) / 2
        ref_y = (left_shoulder.y + right_shoulder.y) / 2
        ref_z = (left_shoulder.z + right_shoulder.z) / 2
        pose = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z]
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 3)

    if results.left_hand_landmarks and results.pose_landmarks:
        lh = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z]
                       for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks and results.pose_landmarks:
        rh = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z]
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])

def Z_score_normalization(data):
    """
    Chuẩn hóa dữ liệu bằng phương pháp Z-score.
    """
    return (data - mean) / std

@api.route('/recognize', methods=['POST'])
def recognize_sign_language():
    try:
        # Kiểm tra xem request có chứa ảnh và frame_id không
        data = request.json
        image_data = data.get('image')
        frame_id = data.get('frame_id')

        if not image_data or frame_id is None:
            return jsonify({'error': 'No image data or frame_id provided'}), 400

        # Giải mã ảnh base64
        image_data = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Resize ảnh để phù hợp với Mediapipe
        image = cv2.resize(image, (224, 224))

        # Mediapipe detection
        _, results = mediapipe_detection(image, holistic_model)

        # Kiểm tra kết quả Mediapipe
        if results is None or not results.pose_landmarks:
            error_response = {
                'error': 'No landmarks detected',
                'frame_id': frame_id
            }
            send_to_web('result', error_response)
            return jsonify(error_response), 400

        # Trích xuất keypoints
        keypoints = extract_keypoints(results)

        # Chuẩn hóa keypoints
        keypoints = Z_score_normalization(np.array([keypoints]))

        # Thêm chiều batch size
        keypoints = np.expand_dims(keypoints, axis=0)

        # Dự đoán bằng model
        outputs = model.predict(keypoints, verbose=0)
        predicted = np.argmax(outputs, axis=1)
        confidence = float(np.max(outputs))
        label = labels[predicted[0]]

        # Chuẩn bị kết quả
        result = {
            'frame_id': frame_id,
            'recognized_text': label,
            'confidence': confidence,
            'timestamp': int(time.time() * 1000)
        }

        # Gửi kết quả lên web
        send_to_web('result', result)

        # Gửi video stream lên web (resize nhỏ hơn để tiết kiệm băng thông)
        stream_frame = cv2.resize(image, (320, 240))
        _, stream_buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        stream_data = {
            'image': f"data:image/jpeg;base64,{base64.b64encode(stream_buffer).decode('utf-8')}",
            'timestamp': int(time.time() * 1000)
        }
        send_to_web('stream', stream_data)

        # Trả về kết quả cho Raspberry Pi
        return jsonify(result)

    except Exception as e:
        print(f"Error occurred: {e}")
        error_response = {
            'error': str(e),
            'frame_id': frame_id
        }
        send_to_web('result', error_response)
        return jsonify(error_response), 500