import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttention(nn.Module):
    """
    Hierarchical Spatiotemporal Attention (HSTA).
    1. Local Attention: Within each temporal segment.
    2. Global Attention: Across segments to capture long-term context.
    """
    def __init__(self, input_dim):
        super(HierarchicalAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim // 8)
        self.key = nn.Linear(input_dim, input_dim // 8)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1)**0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        return out, attn_weights

class SegmentAttention(nn.Module):
    """Attention within a single segment"""
    def __init__(self, input_dim):
        super(SegmentAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (batch, segment_len, dim)
        weights = self.attn(x)
        out = torch.sum(x * weights, dim=1)
        return out, weights
