from model import ASL_model
from sklearn.model_selection import train_test_split
from dataset import ASL_Dataset
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tracker import MisclassificationTracker

# Đường dẫn đến keypoints
keypoints_path = "D:/2025/ASL_model/keypoints_v2/*/*"
dataset = ASL_Dataset(keypoints_path)
# print(dataset[1][0][1])


label_to_index = {label: idx for idx, label in enumerate(dataset.label_list)}
all_labels = np.array([label_to_index[label] for label in dataset.labels])

indices = np.arange(len(all_labels))
train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, all_labels, test_size=0.3, random_state=42, stratify=all_labels)

val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=temp_labels)

# Tạo tập dữ liệu
train_data = [dataset[i][0] for i in train_idx]
train_labels = [dataset[i][1] for i in train_idx]

val_data = [dataset[i][0] for i in val_idx]
val_labels = [dataset[i][1] for i in val_idx]

test_data = [dataset[i][0] for i in test_idx]
test_labels = [dataset[i][1] for i in test_idx]

train_data = np.array(train_data)
train_labels = np.array(train_labels)

val_data = np.array(val_data)
val_labels = np.array(val_labels)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
np.save("D:/2025/ASL_model/Codev2/mean.npy", mean)
np.save("D:/2025/ASL_model/Codev2/std.npy", std)


train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

# Khởi tạo mô hình
model = ASL_model(dataset[0][0].shape, len(dataset.label_list))

# Biên dịch mô hình
# model.compile(
#     optimizer=Adam(learning_rate=0.0001),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Thiết lập callback
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,              # Giảm LR theo hệ số này
    patience=3,              # Sau 3 epochs không cải thiện thì giảm
    min_lr=1e-6,             # LR tối thiểu
    verbose=15
)

history = model.fit(
    train_data, train_labels,
    validation_data=(val_data, val_labels),
    epochs=120,
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler],
)


# Đánh giá mô hình trên tập test
test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=64)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

model.save("D:/2025/ASL_model/Codev2/model.h5")


def plot_training_history(history, output_dir="D:/2025/ASL_model/plots"):
    """
    Vẽ biểu đồ loss và accuracy từ lịch sử huấn luyện và lưu thành file.

    Args:
        history: Lịch sử huấn luyện trả về từ model.fit().
        output_dir: Thư mục để lưu các biểu đồ.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Vẽ biểu đồ loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # Vẽ biểu đồ accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()

plot_training_history(history)