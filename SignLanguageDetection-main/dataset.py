import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def padding_keypoints(keypoints, max_len=40):
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

class ASL_Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.paths = glob.glob(path)
        self.labels = [str(os.path.basename(os.path.normpath(os.path.dirname(p)))) for p in self.paths]
        self.label_list = sorted(set(self.labels)) 
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        keypoint_path = self.paths[index]
        label_str = self.labels[index]
        label_idx = self.label_list.index(label_str)

        keypoints = np.load(keypoint_path)
        # keypoints = padding_keypoints(keypoints, max_len = 40) 
        keypoints = np.array(keypoints, dtype=np.float32)

        if self.transform:
            keypoints = torch.stack([
                self.transform(Image.fromarray(key)) for key in keypoints
            ])

        label = torch.tensor(label_idx, dtype=torch.long)
        return keypoints, label