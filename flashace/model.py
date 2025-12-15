import torch
import torch.nn as nn
from .physics import ACE_Descriptor
from .attention import DenseFlashAttention

class FlashACE(nn.Module):
    def __init__(self, r_max=5.0, l_max=2, num_radial=8, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim  
        self.r_max = r_max
        self.l_max = l_max
        
        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACE_Descriptor(r_max, l_max, num_radial, hidden_dim)
        
        self.layers = nn.ModuleList([
            DenseFlashAttention(self.ace.irreps_out, hidden_dim) for _ in range(num_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.SiLU(), 
            nn.Linear(64, 1)
        )

    def forward(self, data, training=False):
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        cell_volume = data.get('volume', None)

        # We always need gradients w.r.t. atomic positions to compute forces.
        # Detach to ensure we work with a leaf tensor before enabling grads.
        pos = pos.detach()
        pos.requires_grad_(True)

        if training and cell_volume is not None:
            epsilon = torch.zeros(3, 3, device=pos.device, requires_grad=True)
            pos = pos @ (torch.eye(3, device=pos.device) + epsilon)
        else:
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_len = torch.norm(edge_vec, dim=1)
        
        # 1. Pipeline (No checkpoints)
        h = self.emb(z)
        h = self.ace(h, edge_index, edge_vec, edge_len)
        for layer in self.layers: 
            h = layer(h, edge_index)
            
        # 2. Readout
        # Note: We extract only the scalar (L=0) features for energy
        # The optimized physics.py puts scalars first, so this slice is correct.
        scalars = h[:, :self.hidden_dim] 
        E = torch.sum(self.readout(scalars))
        
        # 3. Derivatives
        # Avoid building second-order graphs during evaluation to reduce memory.
        grad_opts = {
            'create_graph': training,  # only keep graph for higher-order grads when training
            # Retain the graph during training so we can also differentiate w.r.t. strain
            # (epsilon) after computing forces.
            'retain_graph': training and epsilon is not None,
            'allow_unused': True,
        }

        grads = torch.autograd.grad(E, pos, **grad_opts)[0]
        F = -grads if grads is not None else torch.zeros_like(pos)
        
        S = torch.zeros(3, 3, device=pos.device)
        if training and epsilon is not None:
            # Retain the graph so the outer loss.backward() can still traverse
            # the computation graph built when taking the strain derivative.
            g_eps = torch.autograd.grad(
                E,
                epsilon,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g_eps is not None: S = -g_eps / cell_volume
                
        return E, F, S
