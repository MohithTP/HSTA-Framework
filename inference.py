import torch
import cv2
import numpy as np
import argparse
from models.hybrid_summarizer import HSTA_Summarizer
from feature_extractor import FeatureExtractor
from data_loader import get_video_segments

def generate_summary(video_path, model_path, output_path="summary.mp4", top_k_percent=0.15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing on {device}")

    # 1. Load Model
    model = HSTA_Summarizer().to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded.")
    else:
        print("Warning: No model found, using random initialization for demo.")

    model.eval()

    # 2. Extract Features
    print("Extracting features...")
    # For real inference, we need to process the video frame by frame or batch of frames
    # Our data loader splits into segments. 
    # Let's manually handle this to map back to frame indices.
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames at {fps} fps")
    
    extractor = FeatureExtractor(device=device)
    
    frames = []
    frame_indices = []
    
    # Process in batches for memory efficiency
    batch_size = 32
    current_batch = []
    
    all_features = []
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_batch.append(frame_rgb)
        
        if len(current_batch) == batch_size:
            feats = extractor.extract(current_batch)
            all_features.append(feats)
            current_batch = []
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} frames", end='\r')

    if current_batch:
        feats = extractor.extract(current_batch)
        all_features.append(feats)
        
    cap.release()
    
    # Concatenate all features: (Total_Frames, 960)
    if not all_features:
        print("No features extracted.")
        return

    features_tensor = torch.cat(all_features)
    print(f"\nFeatures shape: {features_tensor.shape}")
    
    # Reshape into segments for the model (N, 80, 960)
    segment_size = 80
    num_segments = int(np.ceil(features_tensor.shape[0] / segment_size))
    
    # Pad features
    pad_len = num_segments * segment_size - features_tensor.shape[0]
    if pad_len > 0:
        padding = torch.zeros(pad_len, features_tensor.shape[1])
        features_padded = torch.cat([features_tensor, padding])
    else:
        features_padded = features_tensor
        
    # Reshape: (Num_Segments, Segment_Len, Dim)
    segments = features_padded.view(num_segments, segment_size, -1).to(device)
    
    # 3. Model Inference
    with torch.no_grad():
        scores_seg, _, _ = model(segments) # (Num_Segments, Segment_Len)
        
    # Flatten scores and remove padding
    scores = scores_seg.view(-1).cpu().numpy()
    scores = scores[:total_frames]
    
    # 4. Select Keyframes
    # Strategy: Select top K% frames or use a threshold. 
    # For video summary, we often want contiguous clips.
    # Simple approach: Select frames with score > threshold, then group them.
    
    # Let's use simple top K selection for now
    num_summary_frames = int(total_frames * top_k_percent)
    top_indices = np.argsort(scores)[::-1][:num_summary_frames]
    top_indices = sorted(top_indices) # Temporal order
    
    print(f"Selected {len(top_indices)} frames for summary.")
    
    # 5. Generate Summary Video
    # To make it watchable, we might want to include neighbors of high-score frames
    # But for "stitched keyframes" demo:
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    summary_idx = 0
    
    # Optimization: Read only needed frames? Seeking is slow.
    # Linear scan is better if summary is dense.
    
    top_indices_set = set(top_indices)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in top_indices_set:
            out.write(frame)
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Summary video saved to {output_path}")

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--model", type=str, default="models/hsta_model.pth", help="Path to trained model")
    args = parser.parse_args()
    
    generate_summary(args.input, args.model)
