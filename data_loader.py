import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    """
    Dataset class for video frames. 
    Can either load pre-extracted features or raw frames.
    For the HSTA novelty, we will segment the video.
    """
    def __init__(self, video_path, segment_size=80, transform=None):
        self.video_path = video_path
        self.segment_size = segment_size
        self.frames = self._load_frames()

    def _load_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        # We return the number of segments
        return int(np.ceil(len(self.frames) / self.segment_size))

    def __getitem__(self, idx):
        start = idx * self.segment_size
        end = min((idx + 1) * self.segment_size, len(self.frames))
        segment = self.frames[start:end]
        
        # Padding if segment is shorter than segment_size
        if len(segment) < self.segment_size:
            pad_size = self.segment_size - len(segment)
            last_frame = segment[-1]
            segment.extend([last_frame] * pad_size)
            
        return np.array(segment)

def get_video_segments(video_path, segment_size=80):
    dataset = VideoDataset(video_path, segment_size=segment_size)
    return dataset

if __name__ == "__main__":
    # Test with a dummy check (replace with actual path if needed)
    print("VideoDataset class initialized.")
