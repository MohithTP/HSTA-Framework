import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from models.hybrid_summarizer import HSTA_Summarizer
from models.feature_extractor import FeatureExtractor
from .audio_extractor import AudioExtractor

class VideoSummarizer:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = FeatureExtractor(device=self.device)
        self.audio_extractor = AudioExtractor()
        
        # Input dim: 960 (Visual) + 128 (Audio) = 1088
        self.input_dim = 1088
        self.model = HSTA_Summarizer(input_dim=self.input_dim).to(self.device)
        
        # Load weights if exist
        if os.path.exists("models/hsta_multimodal.pth"):
            self.model.load_state_dict(torch.load("models/hsta_multimodal.pth", map_location=self.device))
        elif os.path.exists("models/hsta_model.pth"):
            try:
                self.model.load_state_dict(torch.load("models/hsta_model.pth", map_location=self.device))
            except RuntimeError:
                print("Warning: Architecture mismatch (Visual vs Multimodal). Using random initialization.")
        
        self.model.eval()
        self.blip_processor = None
        self.blip_model = None

    def load_blip(self):
        if self.blip_processor is None:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def summarize(self, video_path, summary_ratio=0.15, status_callback=None):
        if status_callback: status_callback("Initializing Feature Extraction...")
        
        # 1. Extract Audio Features first
        if status_callback: status_callback("Extracting Audio (MFCCs)...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract audio for the whole video
        skip_frames = 15
        effective_fps = fps / skip_frames
        
        # Extract audio with new effective FPS
        audio_features = self.audio_extractor.extract(video_path, fps=effective_fps)
        
        # Ensure length matches total subsampled frames (truncate or pad)
        expected_len = int(total_frames / skip_frames) + (1 if total_frames % skip_frames != 0 else 0)
        
        if len(audio_features) > expected_len:
            audio_features = audio_features[:expected_len]
        elif len(audio_features) < expected_len:
            pad = np.zeros((expected_len - len(audio_features), 128))
            audio_features = np.vstack((audio_features, pad))
            
        audio_features = torch.tensor(audio_features, dtype=torch.float32)

        # 2. Extract Visual Features
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_buffer = []
        features_list = []
        
        curr_frame_idx = 0 # Index in the subsampled feature space
        raw_frame_idx = 0  # Absolute frame index
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if raw_frame_idx % skip_frames == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_buffer.append(img)
                
                if len(frames_buffer) >= 32:
                    if status_callback: status_callback(f"Extracting Visual+Audio Features: {int(raw_frame_idx/total_frames*100)}%")
                    
                    # Visual Features (32, 960)
                    vis_feats = self.extractor.extract(frames_buffer)
                    
                    # Audio Features (32, 128)
                    end_idx = curr_frame_idx + len(frames_buffer)
                    # Safe slice with padding if needed
                    if end_idx > len(audio_features):
                        # Pad audio if visual is slightly ahead due to rounding
                        pad_size = end_idx - len(audio_features)
                        aud_pad = torch.zeros(pad_size, 128)
                        aud_feats = torch.cat([audio_features[curr_frame_idx:], aud_pad])
                    else:
                        aud_feats = audio_features[curr_frame_idx : end_idx]
                    
                    # Fuse: (32, 1088)
                    if vis_feats.shape[0] == aud_feats.shape[0]:
                        fused_feats = torch.cat((vis_feats, aud_feats), dim=1)
                        features_list.append(fused_feats)
                    
                    curr_frame_idx += len(frames_buffer)
                    frames_buffer = []
            
            raw_frame_idx += 1
        
        if frames_buffer:
            vis_feats = self.extractor.extract(frames_buffer)
            # Handle remaining audio
            aud_feats = audio_features[curr_frame_idx:]
            # Pad audio to match visual
            if aud_feats.shape[0] < vis_feats.shape[0]:
                pad_size = vis_feats.shape[0] - aud_feats.shape[0]
                aud_feats = torch.cat([aud_feats, torch.zeros(pad_size, 128)])
            elif aud_feats.shape[0] > vis_feats.shape[0]:
                aud_feats = aud_feats[:vis_feats.shape[0]]
                
            fused_feats = torch.cat((vis_feats, aud_feats), dim=1)
            features_list.append(fused_feats)
            
        features = torch.cat(features_list)
        
        # HSTA Inference
        if status_callback: status_callback("HSTA Temporal Modeling (Multimodal)...")
        segment_size = 80
        num_segments = int(np.ceil(features.shape[0] / segment_size))
        pad_len = num_segments * segment_size - features.shape[0]
        if pad_len > 0:
            features = torch.cat([features, torch.zeros(pad_len, features.shape[1])])
        segments = features.view(num_segments, segment_size, -1).to(self.device)
        
        with torch.no_grad():
            scores, _, _ = self.model(segments)
            
        scores = scores.view(-1).cpu().numpy()[:total_frames]
        
        # Normalize scores for better visualization (0 to 1)
        # This fixes the "flat graph" issue if raw scores are small
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Select keyframes
        num_summary_frames = max(1, int(total_frames * summary_ratio))
        top_indices = np.argsort(scores)[-num_summary_frames:]
        top_indices = sorted(top_indices)
        
        # Generate summary video path
        if status_callback: status_callback("Encoding Summary Video...")
        output_path = video_path.replace(".mp4", "_summary.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        curr = 0
        top_set = set(top_indices)
        while True:
            ret, frame = cap.read()
            if not ret: break
            if curr in top_set:
                out.write(frame)
            curr += 1
            
        out.release()
        cap.release()
        
        return output_path, scores, top_indices

    def generate_captions(self, video_path, top_indices, num_captions=3, status_callback=None):
        if status_callback: status_callback("Loading BLIP Semantic Model...")
        self.load_blip()
        cap = cv2.VideoCapture(video_path)
        
        # Limit to top-3 but spacing them out to avoid same-scene redundancy
        unique_indices = sorted(list(set(top_indices)))
        
        step = max(1, len(unique_indices) // num_captions)
        selected_indices = [unique_indices[i] for i in range(0, len(unique_indices), step)][:num_captions]
        
        results = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        curr = 0
        cap_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            if curr in selected_indices:
                if status_callback: status_callback(f"Captioning Fragment {cap_count+1}/{len(selected_indices)}...")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Improved Prompting
                inputs = self.blip_processor(images=rgb_frame, text="A photo of", return_tensors="pt").to(self.device)
                out = self.blip_model.generate(**inputs, max_new_tokens=20)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                
                # Use unique filenames for frames
                task_prefix = os.path.basename(video_path).split('_')[0]
                frame_filename = f"static/temp_frames/{task_prefix}_frame_{curr}.jpg"
                os.makedirs("static/temp_frames", exist_ok=True)
                cv2.imwrite(frame_filename, frame)
                
                results.append({
                    "frame_idx": curr,
                    "caption": caption.replace("a photo of ", "").capitalize(),
                    "image_url": f"/{frame_filename}"
                })
                cap_count += 1
            curr += 1
            
        cap.release()
        return results
