import torch
import torch.nn as nn
from .physics import ACE_Descriptor
from .attention import DenseFlashAttention, LocalMessagePassing


class LocalMPAttentionBlock(nn.Module):
    """Single local message passing + attention block.

    Keeps locality high via distance-decayed aggregation, then lets attention
    share information directionally within the same neighborhood. This keeps
    the stack shallow and fast while preserving short-range force sensitivity.
    """

    def __init__(
        self,
        irreps_in,
        hidden_dim,
        use_local_mp: bool = True,
        local_mp_sharpness: float = 6.0,
        attention_message_clip: float | None = None,
        attention_conditioned_decay: bool = True,
        attention_share_qkv: str | bool = "none",
    ):
        super().__init__()
        self.use_local_mp = use_local_mp
        self.local_mp = (
            LocalMessagePassing(irreps_in, sharpness=local_mp_sharpness)
            if use_local_mp
            else None
        )
        self.attention = DenseFlashAttention(
            irreps_in,
            hidden_dim,
            message_clip=attention_message_clip,
            use_conditioned_decay=attention_conditioned_decay,
            share_qkv_mode=attention_share_qkv,
        )

    def forward(self, h, edge_index, edge_vec, edge_len, temperature_scale: float = 1.0):
        if self.use_local_mp and self.local_mp is not None:
            h = self.local_mp(h, edge_index, edge_len)
        h = self.attention(h, edge_index, edge_vec, edge_len, temperature_scale=temperature_scale)
        return h


class FlashACE(nn.Module):
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
        local_message_passing: bool = True,
        local_mp_sharpness: float = 6.0,
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
        self.use_local_mp = local_message_passing

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
        self.single_block = LocalMPAttentionBlock(
            self.ace.irreps_out,
            hidden_dim,
            use_local_mp=local_message_passing,
            local_mp_sharpness=local_mp_sharpness,
            attention_message_clip=attention_message_clip,
            attention_conditioned_decay=attention_conditioned_decay,
            attention_share_qkv=attention_share_qkv,
        )
        self.blocks = None if num_layers <= 1 else nn.ModuleList(
            [self.single_block] + [
                LocalMPAttentionBlock(
                    self.ace.irreps_out,
                    hidden_dim,
                    use_local_mp=local_message_passing,
                    local_mp_sharpness=local_mp_sharpness,
                    attention_message_clip=attention_message_clip,
                    attention_conditioned_decay=attention_conditioned_decay,
                    attention_share_qkv=attention_share_qkv,
                )
                for _ in range(num_layers - 1)
            ]
        )
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.SiLU(), 
            nn.Linear(64, 1)
        )
        # Layer normalization on scalar channels stabilizes the single-layer
        # regime and makes the force head less sensitive to feature scale drift.
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.force_gate = nn.Linear(1, 1)
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
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        cell_volume = data.get('volume', None)

        # We always need gradients w.r.t. atomic positions to compute forces.
        # Detach to ensure we work with a leaf tensor before enabling grads.
        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if training and cell_volume is not None:
            # Parameterize the small-strain tensor symmetrically so the stress
            # we backpropagate through corresponds to the symmetric Cauchy
            # stress and does not pick up spurious rotational components. This
            # matches how ACE/MACE form stresses by differentiating with
            # respect to symmetric lattice strains.
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
        
        # 1. Pipeline (No checkpoints)
        h = self.emb(z)
        h = self.ace(h, edge_index, edge_vec, edge_len)

        if self.blocks is None:
            h = self.single_block(h, edge_index, edge_vec, edge_len, temperature_scale=temperature_scale)
        else:
            for block in self.blocks:
                h = block(h, edge_index, edge_vec, edge_len, temperature_scale=temperature_scale)
            
        # 2. Readout
        # Note: We extract only the scalar (L=0) features for energy
        # The optimized physics.py puts scalars first, so this slice is correct.
        scalars = h[:, :self.hidden_dim]
        scalars = self.scalar_norm(scalars)
        E = torch.sum(self.readout(scalars))

        aux = {}
        if self.use_aux_force_head and self.aux_force_head is not None:
            mean_edge = edge_len.mean().unsqueeze(0)
            gate = torch.sigmoid(self.force_gate(mean_edge))
            aux['force'] = gate * self.aux_force_head(scalars)
        if self.use_aux_stress_head and self.aux_stress_head is not None:
            pooled = scalars.mean(dim=0, keepdim=True)
            stress_voigt = self.aux_stress_head(pooled).view(-1)
            aux['stress'] = stress_voigt

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
                strain_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                # Map the 6 unique components back to a symmetric stress tensor
                # and normalize by the deformed volume to avoid overestimating
                # stress under volumetric strain.
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
