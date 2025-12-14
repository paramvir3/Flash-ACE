import math

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet


class PolynomialCutoff(nn.Module):
    """Smooth polynomial envelope used in MACE's radial basis.

    The expression ``1 - (p + 1) * x**p + p * x**(p + 1)`` forces both the value and
    first derivative to vanish at the cutoff, matching the formulation in
    mace/modules/radial.py.
    """

    def __init__(self, r_max: float, p: int = 5):
        super().__init__()
        self.r_max = float(r_max)
        self.p = p

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(distances / self.r_max, max=1.0)
        x_p = torch.pow(x, self.p)
        return 1 - (self.p + 1) * x_p + self.p * x_p * x


class BesselBasis(nn.Module):
    """Bessel basis with normalization matching mace/modules/radial.py."""

    def __init__(self, r_max: float, num_radial: int, trainable: bool = False):
        super().__init__()
        self.r_max = float(r_max)
        freq = torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype()) * math.pi
        if trainable:
            self.freq = nn.Parameter(freq)
        else:
            self.register_buffer("freq", freq)
        self.norm = math.sqrt(2 / self.r_max)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = torch.clamp(distances, min=0.0)
        scaled = distances.unsqueeze(-1) * self.freq / self.r_max

        # Use the analytical limit for r -> 0 to avoid NaNs.
        safe_dist = distances.unsqueeze(-1).clamp(min=torch.finfo(distances.dtype).eps)
        bessel = torch.sin(scaled) / safe_dist
        bessel = torch.where(
            distances.unsqueeze(-1) == 0,
            self.freq / self.r_max,
            bessel,
        )
        return self.norm * bessel


class ACERadialBasis(nn.Module):
    """Polynomial cutoff times Bessel basis, mirroring MACE."""

    def __init__(self, r_max: float, num_radial: int, envelope_exponent: int = 5):
        super().__init__()
        self.cutoff = PolynomialCutoff(r_max, p=envelope_exponent)
        self.bessel = BesselBasis(r_max, num_radial)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        cutoff = self.cutoff(distances)
        return cutoff.unsqueeze(-1) * self.bessel(distances)

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
        self.radial_basis = ACERadialBasis(r_max, num_radial)
        # Align spherical harmonics normalization with MACE's implementation
        # and ensure we feed unit vectors during the forward pass.
        self.sh = o3.SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization="integral"
        )

        # Tensor Product (Radial * Angular)
        # Match MACE: distance-dependent weights come from the radial MLP rather
        # than being learned as global parameters of the tensor product.
        self.tp_a = o3.FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.radial_net = FullyConnectedNet(
            [num_radial, 64, self.tp_a.weight_numel], torch.nn.functional.silu
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
        radial_emb = self.radial_basis(edge_len)
        radial_weights = self.radial_net(radial_emb)
        # Normalize edge vectors before evaluating spherical harmonics so the
        # angular part matches the MACE reference implementation.
        normalized_vec = edge_vec / edge_len.clamp_min(torch.finfo(edge_len.dtype).eps).unsqueeze(-1)
        Y_lm = self.sh(normalized_vec)

        # Incorporate atomic attributes on the sending atoms and modulate the
        # tensor-product paths with distance-dependent weights, matching the
        # ACE A-basis construction used inside MACE.
        node_feats = node_attrs[sender]
        edge_feats = self.tp_a(node_feats, Y_lm, radial_weights)

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
