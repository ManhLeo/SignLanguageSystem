import logging
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pandas as pd
import os

logging.basicConfig(level=logging.WARNING)
app = FastAPI()

# Load mô hình và nhãn
MODEL_PATH = "D:/2025/ASL_model/Codev2/model.h5"
LABELS_PATH = "D:/2025/ASL_model/Codev2/labels.csv"
labels = pd.read_csv(LABELS_PATH)['label'].tolist()
model = load_model(MODEL_PATH)

# Khởi tạo Mediapipe Holistic
holistic_model = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def mediapipe_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image)
    return results

def extract_keypoints(results):
    # Trích xuất keypoints: pose (33*3), left hand (21*3), right hand (21*3)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])  # Shape (225,)

def Z_score_normalization(data):
    mean = np.load("D:/2025/ASL_model/Codev2/mean.npy")
    std = np.load("D:/2025/ASL_model/Codev2/std.npy")
    return (data - mean) / std

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    client_host = request.client.host.replace('.', '_')
    save_dir = "D:/2025/ASL_model/Codev2/frame_check"
    file_path = os.path.join(save_dir, f"{client_host}.npy")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Đọc dữ liệu cũ nếu có
    keypoints = []
    if os.path.isfile(file_path):
        keypoints = np.load(file_path).tolist()

    # Đọc ảnh gửi từ client
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))

    # Xử lý ảnh với Mediapipe
    results = mediapipe_detection(img)
    keypoint = extract_keypoints(results)
    keypoints.append(keypoint)
    keypoints = keypoints[-30:]  # Giữ 30 frame gần nhất

    # Lưu lại frame keypoints mới
    np.save(file_path, np.array(keypoints))

    # Nếu đủ 30 frame thì dự đoán
    if len(keypoints) == 30:
        keypoints_np = np.array(keypoints)  # Shape (30, 225)
        keypoints_np = Z_score_normalization(keypoints_np)
        outputs = model.predict(np.array([keypoints_np]), verbose=0)  # Input shape (1, 30, 225)
        predicted = np.argmax(outputs, axis=1)
        confidence = float(np.max(outputs))
        label = labels[predicted[0]]
        print(f"Dự đoán: {label} (Độ tự tin: {confidence:.2f}%)")

        return JSONResponse(
            status_code=200,
            content={"label": label, "confidence": confidence}
        )

    return JSONResponse(
        status_code=200,
        content={"label": "...", "confidence": 0.0}
    )


# uvicorn main:app --reload
