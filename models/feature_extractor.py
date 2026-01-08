import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    """
    Extracts spatial features from video frames using a pre-trained MobileNetV3.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.device = device
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier = nn.Identity() 
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, frame_list):
        """
        frame_list: List of PIL Images or numpy arrays
        returns: torch.Tensor of shape (N, 960) for MobileNetV3-Large
        """
        batch = []
        for frame in frame_list:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            batch.append(self.transform(frame))
        
        batch_tensor = torch.stack(batch).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch_tensor)
        
        return features.cpu()

if __name__ == "__main__":
    # Quick test
    extractor = FeatureExtractor()
    dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    feats = extractor.extract(dummy_frames)
    print(f"Extracted features shape: {feats.shape}") # Should be (5, 960)
