import os
import glob 
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

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
    # Lấy keypoints của pose (33 điểm)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)

    # Lấy keypoints của bàn tay trái (21 điểm)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Lấy keypoints của bàn tay phải (21 điểm)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    # Kết hợp tất cả keypoints
    return np.concatenate([pose, lh, rh])

# def extract_keypoints(results):
#     if results.pose_landmarks:
#         # Lấy tọa độ hai vai
#         left_shoulder = results.pose_landmarks.landmark[11]  # Left shoulder
#         right_shoulder = results.pose_landmarks.landmark[12]  # Right shoulder
#         # Tính trung điểm
#         ref_x = (left_shoulder.x + right_shoulder.x) / 2
#         ref_y = (left_shoulder.y + right_shoulder.y) / 2
#         ref_z = (left_shoulder.z + right_shoulder.z) / 2
#         # Chuẩn hóa pose landmarks
#         pose = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z] 
#                         for res in results.pose_landmarks.landmark]).flatten()
#     else:
#         pose = np.zeros(33 * 3)
#         ref_x, ref_y, ref_z = 0, 0, 0

#     # Chuẩn hóa tay trái
#     if results.left_hand_landmarks and results.pose_landmarks:
#         lh = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z] 
#                       for res in results.left_hand_landmarks.landmark]).flatten()
#     else:
#         lh = np.zeros(21 * 3)

#     # Chuẩn hóa tay phải
#     if results.right_hand_landmarks and results.pose_landmarks:
#         rh = np.array([[res.x - ref_x, res.y - ref_y, res.z - ref_z] 
#                       for res in results.right_hand_landmarks.landmark]).flatten()
#     else:
#         rh = np.zeros(21 * 3)

#     return np.concatenate([pose, lh, rh])

def padding_keypoints(keypoints, max_len=30):
    if len(keypoints) < max_len:
        while len(keypoints) < max_len:
            keypoints.insert(0, np.zeros_like(keypoints[0]))
            if len(keypoints) < max_len:
                keypoints.append(np.zeros_like(keypoints[0]))
        return keypoints
    
    padding_keypoints = []
    if len(keypoints) > max_len:
        interval = len(keypoints) // max_len
        for i in range(max_len):
            padding_keypoints.append(keypoints[i*interval])
        return padding_keypoints
    
    return keypoints

def data_preprocessing(video_path, model):
    video = cv2.VideoCapture(video_path)
    keypoints_list = []
    interval = (video.get(cv2.CAP_PROP_FPS) + 29) // 30 
    count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frame, results= mediapipe_detection(frame, model)

            keypoints = extract_keypoints(results)
            keypoints_list.append(keypoints)

        count += 1
    video.release()
    
    return np.array(keypoints_list)
holistic_model = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def preprocess_data(input_path, output_path, model, max_len=30):
    """
    Tiền xử lý dữ liệu video và lưu keypoints vào file numpy.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Lấy danh sách tất cả các video trong thư mục con
    video_paths = glob.glob(os.path.join(input_path, "**", "*.mp4"), recursive=True) + \
                  glob.glob(os.path.join(input_path, "**", "*.MOV"), recursive=True)
    print(f"Found {len(video_paths)} videos.")

    for video in tqdm(video_paths, desc="Processing videos"):
        filename = os.path.basename(video)
        relative_path = os.path.relpath(video, input_path)  
        label_dir = os.path.dirname(relative_path)  
        output_label_dir = os.path.join(output_path, label_dir) 

        try:
            keypoints = data_preprocessing(video, model)
            keypoints = padding_keypoints(np.array(keypoints), max_len=max_len)  
            output_file = os.path.join(output_label_dir, filename.replace(".MOV", ".npy").replace(".mp4", ".npy"))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.save(output_file, keypoints)
        except Exception as e:
            print(f"Error processing video {video}: {e}")

# Khởi tạo Mediapipe Holistic
holistic_model = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Đường dẫn đầu vào và đầu ra
input_path = "D:/2025/ASL_model/datav2"
output_path = "D:/2025/ASL_model/keypoints_v2"

# Chạy tiền xử lý
preprocess_data(input_path, output_path, holistic_model)
print("Preprocessing completed.")