import torch
import torch.optim as optim
import torch.nn as nn
from models.hybrid_summarizer import HSTA_Summarizer
from models.feature_extractor import FeatureExtractor
from utils.audio_extractor import AudioExtractor
import cv2
import numpy as np
import os
import argparse

def extract_features(video_path, device):
    print(f"Extracting features for {video_path}...")
    
    # 1. Audio
    audio_ext = AudioExtractor()
    # Get FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    audio_feats = audio_ext.extract(video_path, fps=fps) # (T, 128)
    
    # 2. Visual
    vis_ext = FeatureExtractor(device=device)
    frames_buffer = []
    vis_feats_list = []
    
    curr = 0
    curr = 0
    skip_frames = 15 # Downsample to ~2 FPS (assuming 30fps source)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if curr % skip_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_buffer.append(frame)
            
            if len(frames_buffer) >= 32:
                vf = vis_ext.extract(frames_buffer)
                vis_feats_list.append(vf)
                frames_buffer = []
        curr += 1
            
    if frames_buffer:
        vf = vis_ext.extract(frames_buffer)
        vis_feats_list.append(vf)
        
    cap.release()
    
    if not vis_feats_list:
        print("Error: No frames extracted.")
        return torch.zeros(1, 1088).to(device)

    vis_feats = torch.cat(vis_feats_list) # (T_subsampled, 960)
    
    # Adjust Audio to match subsampled length
    # Effective FPS for audio sync
    effective_fps = fps / skip_frames
    audio_feats = audio_ext.extract(video_path, fps=effective_fps) # (T_sub, 128)
    
    # 3. Fuse
    # Truncate to min length or pad
    min_len = min(vis_feats.shape[0], audio_feats.shape[0])
    vis_feats = vis_feats[:min_len]
    audio_feats = torch.tensor(audio_feats[:min_len], dtype=torch.float32)
    # If using skip_frames, visual might be slightly less than audio or vice versa.
    # The min_len policy is safe.
    
    fused_feats = torch.cat((vis_feats, audio_feats), dim=1) # (T, 1088)
    
    return fused_feats.to(device)

def train_batch(video_dir, limit=20, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Identify Videos
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return
        
    selected_videos = video_files[:limit]
    print(f"Found {len(video_files)} videos. Training on first {len(selected_videos)}: {selected_videos}")
    
    # 2. Extract & Aggregate Features
    all_segments = []
    
    model = HSTA_Summarizer(input_dim=1088).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    print("Phase 1: Feature Extraction...")
    for i, vid_file in enumerate(selected_videos):
        path = os.path.join(video_dir, vid_file)
        print(f"[{i+1}/{len(selected_videos)}] Processing {vid_file}...")
        try:
            feats = extract_features(path, device) # (T, 1088)
            
            # Segment
            segment_size = 80
            num_segments = int(np.ceil(feats.shape[0] / segment_size))
            pad_len = num_segments * segment_size - feats.shape[0]
            if pad_len > 0:
                feats = torch.cat([feats, torch.zeros(pad_len, 1088).to(device)])
            
            # (Segs, 80, 1088)
            segs = feats.view(num_segments, segment_size, -1)
            all_segments.append(segs)
            
        except Exception as e:
            print(f"Error processing {vid_file}: {e}")
            continue

    if not all_segments:
        print("No valid features extracted.")
        return
    
    print("Phase 2: Training...")
    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        for vid_segments in all_segments:
            scores, _, _ = model(vid_segments)
            scores = scores.view(-1)
            
            loss_sparsity = (scores.mean() - 0.15)**2
            loss_diversity = -torch.var(scores)
            
            loss = loss_sparsity + loss_diversity
            loss.backward()
            
            epoch_loss += loss.item()
        
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(all_segments):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/hsta_multimodal.pth")
    print("Batch training complete. Saved to models/hsta_multimodal.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--folder", type=str, help="Folder path for batch training")
    parser.add_argument("--limit", type=int, default=20, help="Max videos to use")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    if args.folder:
        if os.path.exists(args.folder):
            train_batch(args.folder, args.limit, args.epochs)
        else:
            print(f"Folder not found: {args.folder}")
    elif args.video:
        if os.path.exists(args.video):
            # Reuse batch logic with 1 video
            train_batch(os.path.dirname(args.video), limit=1, epochs=args.epochs) 
        else:
            print(f"Video not found: {args.video}")
    else:
        print("Please provide --video or --folder")
