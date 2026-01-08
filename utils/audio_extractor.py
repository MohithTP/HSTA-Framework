import librosa
import numpy as np
from moviepy import VideoFileClip
import os

class AudioExtractor:
    def __init__(self, target_sr=22050, n_mfcc=128):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc

    def extract(self, video_path, fps=None):
        """
        Extracts MFCC features from the video's audio, synchronized to the video FPS.
        Returns: numpy array of shape (num_frames, n_mfcc)
        """
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                return np.zeros((int(video.duration * (fps or video.fps)), self.n_mfcc))
            
            audio_path = video_path.replace(".mp4", "_temp.wav")
            video.audio.write_audiofile(audio_path, logger=None, fps=self.target_sr)
            
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            video_fps = fps if fps else video.fps
            if not video_fps: video_fps = 30.0
            
            hop_length = int(sr / video_fps)
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=hop_length)
            mfccs = mfccs.T
            
            return mfccs
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            # Return empty features if failure
            return np.zeros((1, self.n_mfcc))
