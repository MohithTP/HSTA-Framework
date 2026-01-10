import torch
import numpy as np
import os

def compute_fscore(pred_scores, gt_scores, overlap_threshold=0.5):
    pred_binary = (pred_scores > 0.5).astype(int)
    gt_binary = (gt_scores > 0.5).astype(int)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    precision = intersection / (pred_binary.sum() + 1e-6)
    recall = intersection / (gt_binary.sum() + 1e-6)
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    f_scores = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            scores, _, _ = model(inputs)
            f = compute_fscore(scores.cpu().numpy().flatten(), targets.cpu().numpy().flatten())
            f_scores.append(f)
    
    avg_f = np.mean(f_scores) if f_scores else 0.0
    print(f"Average F-score: {avg_f:.4f}")
    return avg_f

if __name__ == "__main__":
    from models.hybrid_summarizer import HSTA_Summarizer
    from feature_extractor import FeatureExtractor
    import cv2
    import time
    import glob
    import h5py

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Evaluation on {device}...")
    
    # Paths
    DATA_DIR = "data"
    MAT_PATH = os.path.join(DATA_DIR, "ydata-tvsum50.mat")
    
    # 0. Load Ground Truth
    gt_data = {}
    if os.path.exists(MAT_PATH):
        print(f"Loading Ground Truth from {MAT_PATH}...")
        try:
            with h5py.File(MAT_PATH, 'r') as f:
                data = f['tvsum50']
                video_refs = data['video'][:]
                gt_refs = data['gt_score'][:]
                
                for i in range(len(video_refs)):
                    # Decode ID
                    ref = video_refs[i][0]
                    vid_id = ''.join(chr(c) for c in f[ref][:].flatten()).strip()
                    
                    # Store GT (normalize to 0-1)
                    ref_gt = gt_refs[i][0]
                    scores = f[ref_gt][:].flatten()
                    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
                    gt_data[vid_id] = scores
            print(f"Loaded {len(gt_data)} Ground Truth annotations.")
        except Exception as e:
            print(f"Error loading MAT file: {e}")
    
    # 1. Initialize Model & Extractor
    model = HSTA_Summarizer().to(device)
    if os.path.exists("models/hsta_model.pth"):
        model.load_state_dict(torch.load("models/hsta_model.pth", map_location=device))
        print("Loaded trained model.")
    else:
        print("Using random weights (untrained) - Results will be random.")
        
    extractor = FeatureExtractor(device=device)
    
    # 2. Get Video List
    video_files = glob.glob(os.path.join(DATA_DIR, "*.mp4"))
    print(f"Found {len(video_files)} videos in {DATA_DIR}")
    
    if len(video_files) == 0:
        print("No videos found! Please check the path.")
        exit()
        
    # 3. Run Evaluation Loop
    total_time = 0
    total_frames = 0
    f_scores = []
    
    print(f"{'Video':<20} | {'Frames':<8} | {'FPS':<8} | {'F-Score':<8}")
    print("-" * 65)
    
    try:
        for vid_path in video_files:
            vid_name = os.path.basename(vid_path)
            vid_id = os.path.splitext(vid_name)[0]
            
            # A. Extract Features
            start_time = time.time()
            
            cap = cv2.VideoCapture(vid_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Frame Extraction Loop (Simplified for Speed)
            frames_buffer = []
            features_list = []
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Preprocess & Batch (Naive)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(img)
                
                if len(frames_buffer) >= 32: # Batch size
                    feats = extractor.extract(frames_buffer)
                    features_list.append(feats)
                    frames_buffer = []
            
            if frames_buffer:
                feats = extractor.extract(frames_buffer)
                features_list.append(feats)
                
            cap.release()
            
            if not features_list:
                continue
                
            features = torch.cat(features_list)
            
            # B. Model Inference
            segment_size = 80
            num_segments = int(np.ceil(features.shape[0] / segment_size))
            pad_len = num_segments * segment_size - features.shape[0]
            if pad_len > 0:
                features = torch.cat([features, torch.zeros(pad_len, features.shape[1])])
            segments = features.view(num_segments, segment_size, -1).to(device)
            
            with torch.no_grad():
                scores, _, _ = model(segments)
                
            scores = scores.view(-1).cpu().numpy()[:length]
            
            end_time = time.time()
            duration = end_time - start_time
            proc_fps = length / (duration + 1e-6)
            
            # C. F-Score Calculation
            f_score_val = 0.0
            if vid_id in gt_data:
                gt = gt_data[vid_id]
                # Resize if mismatch (frame dropping issues etc)
                if len(gt) != len(scores):
                    # Simple interpolation to match GT length
                    scores = np.interp(np.linspace(0, 1, len(gt)), np.linspace(0, 1, len(scores)), scores)
                
                f_score_val = compute_fscore(scores, gt)
                f_scores.append(f_score_val)
                f_str = f"{f_score_val*100:.1f}%"
            else:
                f_str = "N/A"
            
            print(f"{vid_name[:20]:<20} | {length:<8} | {proc_fps:<8.1f} | {f_str:<8}")
            
            total_time += duration
            total_frames += length
            
    except KeyboardInterrupt:
        print("\nStopped by user.")
        
    print("-" * 65)
    if total_time > 0:
        avg_fps = total_frames / total_time
        avg_f = np.mean(f_scores) * 100 if f_scores else 0.0
        print(f"Total Processed: {total_frames} frames")
        print(f"Average Speed: {avg_fps:.2f} FPS")
        print(f"Average F-Score: {avg_f:.2f}%")
        
        # Save results
        with open("benchmark_results.txt", "w") as f:
            f.write(f"TVSum Benchmark Results\n")
            f.write(f"Average Speed: {avg_fps:.2f} FPS\n")
            f.write(f"Average F-Score: {avg_f:.2f}%\n")
    else:
        print("No frames processed.")
