import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter_sum, scatter_max

class DenseFlashAttention(nn.Module):
    def __init__(self, irreps_in, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.w_q = o3.Linear(irreps_in, irreps_in)
        self.w_k = o3.Linear(irreps_in, irreps_in)
        self.w_v = o3.Linear(irreps_in, irreps_in)
        self.w_out = o3.Linear(irreps_in, irreps_in)

    def forward(self, x, edge_index):
        sender, receiver = edge_index
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Invariant Dot Product (Scalar Score)
        scores = torch.sum(Q[receiver] * K[sender], dim=1) * self.scale
        
        # Stable Softmax via Scatter
        scores_max, _ = scatter_max(scores, receiver, dim=0, dim_size=x.shape[0])
        alpha = torch.exp(scores - scores_max[receiver])
        Z = scatter_sum(alpha, receiver, dim=0, dim_size=x.shape[0])
        alpha = alpha / (Z[receiver] + 1e-6)
        
        # Aggregate
        weighted_V = alpha.unsqueeze(-1) * V[sender]
        out = scatter_sum(weighted_V, receiver, dim=0, dim_size=x.shape[0])
        
        return x + self.w_out(out)
