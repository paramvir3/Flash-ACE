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

    def forward(self, data, training=False, temperature_scale: float = 1.0, detach_pos: bool = True):
        z, pos, edge_index = data["z"], data["pos"], data["edge_index"]
        cell_volume = data.get("volume", None)

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
        h = self.ace(h, edge_index, edge_vec, edge_len)
        for layer in self.attention_layers:
            h = layer(h, edge_index, edge_vec, edge_len, temperature_scale=temperature_scale)

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
