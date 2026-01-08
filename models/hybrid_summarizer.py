import torch
import torch.nn as nn
from .attention_module import HierarchicalAttention, SegmentAttention

class HSTA_Summarizer(nn.Module):
    """
    Hybrid CNN-RNN Video Summarizer with Hierarchical Spatiotemporal Attention.
    Architecture:
    CNN Features -> Bi-LSTM (Local) -> Segment Attention -> Bi-LSTM (Global) -> Global Attention -> MLP
    """
    def __init__(self, input_dim=960, hidden_dim=256):
        super(HSTA_Summarizer, self).__init__()
        
        self.local_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.local_attn = SegmentAttention(hidden_dim * 2)
        
        self.global_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.global_attn = HierarchicalAttention(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, segments):
        """
        segments: (num_segments, segment_len, input_dim)
        """
        local_out, _ = self.local_lstm(segments) # (num_segments, segment_len, hidden*2)
        segment_summaries, local_weights = self.local_attn(local_out) # (num_segments, hidden*2)
        
        segment_summaries = segment_summaries.unsqueeze(0) # (1, num_segments, hidden*2)
        global_out, _ = self.global_lstm(segment_summaries)
        global_out, global_weights = self.global_attn(global_out) # (1, num_segments, hidden*2)
        
        scores = self.classifier(local_out) # (num_segments, segment_len, 1)
        
        return scores.squeeze(-1), local_weights, global_weights

if __name__ == "__main__":
    model = HSTA_Summarizer()
    dummy_input = torch.randn(5, 80, 960) # 5 segments, 80 frames each
    scores, lw, gw = model(dummy_input)
    print(f"Scores shape: {scores.shape}") # (5, 80)
    print(f"Global weights shape: {gw.shape}")
