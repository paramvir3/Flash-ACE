import math

import torch
import torch.nn as nn

from .attention import DenseFlashAttention
from .physics import ACE_Descriptor


class FlashACE(nn.Module):
    """Flash-ACE with attention-only refinement over ACE descriptors."""

    def __init__(
        self,
        r_max=5.0,
        l_max=2,
        num_radial=8,
        hidden_dim=128,
        num_layers=1,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        attention_message_clip: float | None = None,
        attention_conditioned_decay: bool = True,
        attention_share_qkv: str | bool = "none",
        use_aux_force_head: bool = True,
        use_aux_stress_head: bool = True,
        reciprocal_shells: int = 0,
        reciprocal_scale: float = 1.0,
        reciprocal_pe: bool = False,
        reciprocal_pe_dim: int = 8,
        debye_init: float = 1.0,
        long_range_heads: int = 1,
        long_range_mix: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.l_max = l_max
        self.attention_message_clip = attention_message_clip
        self.attention_conditioned_decay = attention_conditioned_decay
        self.attention_share_qkv = attention_share_qkv
        self.use_aux_force_head = use_aux_force_head
        self.use_aux_stress_head = use_aux_stress_head
        self.reciprocal_shells = max(0, int(reciprocal_shells))
        self.reciprocal_scale = float(reciprocal_scale)
        self.reciprocal_bins = (2 * self.reciprocal_shells + 1) ** 3 - 1 if self.reciprocal_shells > 0 else 0
        self.reciprocal_pe = reciprocal_pe
        self.reciprocal_pe_dim = max(0, int(reciprocal_pe_dim))
        self.recip_pe_proj = nn.Linear(2 * self.reciprocal_pe_dim, hidden_dim) if self.reciprocal_pe and self.reciprocal_pe_dim > 0 else None

        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACE_Descriptor(
            r_max,
            l_max,
            num_radial,
            hidden_dim,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            envelope_exponent=envelope_exponent,
            gaussian_width=gaussian_width,
        )

        self.attention_layers = nn.ModuleList(
            [
                DenseFlashAttention(
                    self.ace.irreps_out,
                    hidden_dim,
                    message_clip=attention_message_clip,
                    use_conditioned_decay=attention_conditioned_decay,
                    share_qkv_mode=attention_share_qkv,
                    long_range_bins=self.reciprocal_bins,
                    debye_init=debye_init,
                    long_range_heads=long_range_heads,
                    long_range_mix=long_range_mix,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.aux_force_head = None
        self.aux_stress_head = None
        if self.use_aux_force_head:
            self.aux_force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, 3),
            )
        if self.use_aux_stress_head:
            self.aux_stress_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 6),
            )

    def _reciprocal_vectors(self, cell):
        if self.reciprocal_shells <= 0 or cell is None:
            return None
        cell_det = torch.det(cell)
        if torch.abs(cell_det) < 1e-8:
            return None
        reciprocal = 2 * math.pi * torch.inverse(cell).t()
        shells = []
        max_n = self.reciprocal_shells
        for h in range(-max_n, max_n + 1):
            for k in range(-max_n, max_n + 1):
                for l in range(-max_n, max_n + 1):
                    if h == k == l == 0:
                        continue
                    g_cart = h * reciprocal[0] + k * reciprocal[1] + l * reciprocal[2]
                    shells.append(g_cart)
        if not shells:
            return None
        return torch.stack(shells, dim=0)  # (M, 3)

    def _reciprocal_features(self, pos, cell):
        G = self._reciprocal_vectors(cell)
        if G is None:
            return None
        phases = pos @ G.t()  # (N, M)
        s_real = torch.cos(phases).sum(dim=0)
        s_imag = torch.sin(phases).sum(dim=0)
        s_mag = torch.sqrt(s_real**2 + s_imag**2 + 1e-9)
        s_norm = s_mag / (pos.shape[0] + 1e-6)
        return s_norm * self.reciprocal_scale

    def forward(self, data, training=False, temperature_scale: float = 1.0, detach_pos: bool = True):
        z, pos, edge_index = data["z"], data["pos"], data["edge_index"]
        cell_volume = data.get("volume", None)
        cell = data.get("cell", None)

        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if training and cell_volume is not None:
            strain_params = torch.zeros(6, device=pos.device, requires_grad=True)
            epsilon = torch.zeros(3, 3, device=pos.device)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]
            deformation = torch.eye(3, device=pos.device) + epsilon
            pos = pos @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_len = torch.clamp(torch.norm(edge_vec, dim=1), max=self.r_max)

        h = self.emb(z)
        G = self._reciprocal_vectors(cell) if self.reciprocal_pe else None
        if self.reciprocal_pe and self.reciprocal_pe_dim > 0 and G is not None:
            g_use = G[: self.reciprocal_pe_dim]
            if g_use.numel() > 0:
                phases = pos @ g_use.t()  # (N, pe_dim)
                sincos = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)  # (N, 2*pe_dim)
                pe_embed = self.recip_pe_proj(sincos)
                h = h + pe_embed
        h = self.ace(h, edge_index, edge_vec, edge_len)
        ace_base = h
        recip = self._reciprocal_features(pos, cell)
        recip_bias = None
        if recip is not None and edge_len.numel() > 0:
            recip_bias = recip.unsqueeze(0).expand(edge_len.shape[0], -1)
        for layer in self.attention_layers:
            # Reinject the static ACE descriptor each layer to mimic iterative
            # message passing over shared geometric features.
            h = h + ace_base
            h = layer(h, edge_index, edge_vec, edge_len, temperature_scale=temperature_scale, reciprocal_bias=recip_bias)

        scalars = self.scalar_norm(h[:, : self.hidden_dim])
        E = torch.sum(self.readout(scalars))

        aux = {}
        if self.use_aux_force_head and self.aux_force_head is not None:
            aux["force"] = self.aux_force_head(scalars)
        if self.use_aux_stress_head and self.aux_stress_head is not None:
            pooled = scalars.mean(dim=0, keepdim=True)
            stress_voigt = self.aux_stress_head(pooled).view(-1)
            aux["stress"] = stress_voigt

        grad_opts = {
            "create_graph": training,
            "retain_graph": training and epsilon is not None,
            "allow_unused": True,
        }
        grads = torch.autograd.grad(E, pos, **grad_opts)[0]
        F = -grads if grads is not None else torch.zeros_like(pos)

        S = torch.zeros(3, 3, device=pos.device)
        if training and epsilon is not None:
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                stress = torch.zeros(3, 3, device=pos.device)
                stress[0, 0] = g_eps[0]
                stress[1, 1] = g_eps[1]
                stress[2, 2] = g_eps[2]
                stress[0, 1] = stress[1, 0] = g_eps[3]
                stress[0, 2] = stress[2, 0] = g_eps[4]
                stress[1, 2] = stress[2, 1] = g_eps[5]
                volume = cell_volume * torch.det(deformation)
                S = -stress / volume

        return E, F, S, aux
