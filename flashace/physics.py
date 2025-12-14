import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace

class ACE_Descriptor(nn.Module):
    def __init__(self, r_max, l_max, num_radial, hidden_dim):
        super().__init__()
        self.r_max = r_max
        self.num_radial = num_radial
        
        # 1. SMART SYMMETRIES (NequIP/ACE Strategy)
        # ------------------------------------------------------------------
        # Instead of giving 'hidden_dim' to everything, we taper it down.
        # Scalars (L=0) get full resolution (Chemistry).
        # Vectors/Tensors (L>0) get reduced resolution (Geometry).
        # This saves massive amounts of memory.
        
        irreps_list = []
        for l in range(l_max + 1):
            if l == 0:
                dim = hidden_dim          # e.g. 128
            elif l == 1:
                dim = hidden_dim // 2     # e.g. 64
            else:
                dim = hidden_dim // 4     # e.g. 32
            
            # Parity (-1)**l is standard
            irreps_list.append((dim, (l, (-1)**l)))
            
        self.irreps_out = o3.Irreps(irreps_list)
        
        # Spherical Harmonics (Geometry Input)
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        
        # Node Features (Scalar Input)
        self.irreps_node = o3.Irreps(f"{hidden_dim}x0e")
        # ------------------------------------------------------------------

        # 2. A-Basis Components
        self.radial_net = FullyConnectedNet([num_radial, 64, hidden_dim], torch.nn.functional.silu)
        self.sh = o3.SphericalHarmonics(self.irreps_sh, normalize=True, normalization='component')
        
        # Tensor Product (Radial * Angular)
        self.tp_a = o3.FullyConnectedTensorProduct(
            self.irreps_node, self.irreps_sh, self.irreps_out,
            internal_weights=True, shared_weights=True
        )
        
        # 3. B-Basis (Symmetric Contraction)
        # Using Elementwise allows us to scale to high L without OOM
        # (This is similar to the ACE 'scaling' approximation)
        self.contraction = o3.ElementwiseTensorProduct(
            self.irreps_out,     # Input 1
            self.irreps_out      # Input 2
        )
        
        # Linear Mixing to recover full feature interactions
        self.mix = o3.Linear(self.contraction.irreps_out, self.irreps_out)

    def forward(self, node_attrs, edge_index, edge_vec, edge_len):
        sender, receiver = edge_index

        # Stage 1: Projection
        radial_emb = soft_one_hot_linspace(
            edge_len, 0.0, self.r_max, self.num_radial,
            basis='bessel', cutoff=True
        )
        R_n = self.radial_net(radial_emb)
        Y_lm = self.sh(edge_vec)

        # Incorporate atomic attributes on the sending atoms and gate them
        # with the learned radial functions before combining with the
        # spherical harmonics. This mirrors the ACE construction used by
        # MACE, where chemical identity enters the A-basis.
        node_feats = node_attrs[sender] * R_n
        edge_feats = self.tp_a(node_feats, Y_lm)

        # Sum Neighbors
        A_basis = torch.zeros(
            node_attrs.shape[0], edge_feats.shape[1],
            device=node_attrs.device, dtype=edge_feats.dtype
        )
        A_basis.index_add_(0, receiver, edge_feats)
        
        # Stage 2: Contraction (Linear scaling with N)
        # B = A * A (Elementwise)
        B_basis = self.contraction(A_basis, A_basis)
        
        # Mix and Add Residual
        return self.mix(B_basis) + A_basis
