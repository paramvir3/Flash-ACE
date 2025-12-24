import torch
import torch.nn as nn
from e3nn import o3
from .physics import ACE_Descriptor
from .attention import DenseFlashAttention

class ScalarMessagePassing(nn.Module):
    """Lightweight, scalar-only message passing to mimic NequIP-style updates."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        scalars, rest = h[..., : self.hidden_dim], h[..., self.hidden_dim :]
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h

        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_len.unsqueeze(-1)], dim=-1)
        msgs = self.mlp(msg_in)
        agg = torch.zeros_like(scalars)
        agg.index_add_(0, receiver, msgs)
        scalars = scalars + agg
        return torch.cat([scalars, rest], dim=-1)

class EdgeUpdate(nn.Module):
    """Per-layer scalar edge update that refreshes node scalars from current states."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        scalars, rest = h[..., : self.hidden_dim], h[..., self.hidden_dim :]
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h

        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_len.unsqueeze(-1)], dim=-1)
        msgs = self.mlp(msg_in)
        agg = torch.zeros_like(scalars)
        agg.index_add_(0, receiver, msgs)
        scalars = scalars + agg
        return torch.cat([scalars, rest], dim=-1)

class EdgeStateInit(nn.Module):
    """Initialize per-edge embeddings from current node scalars and distances."""
    def __init__(self, node_dim: int, edge_state_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, edge_state_dim),
            nn.SiLU(),
            nn.Linear(edge_state_dim, edge_state_dim),
        )

    def forward(self, scalars: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index
        if sender.numel() == 0:
            return torch.zeros((0, self.mlp[-1].out_features), device=scalars.device, dtype=scalars.dtype)
        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_len.unsqueeze(-1)], dim=-1)
        return self.mlp(msg_in)

class EdgeStateUpdate(nn.Module):
    """Update edge embeddings from current node scalars and previous edge state."""
    def __init__(self, node_dim: int, edge_state_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_state_dim + 1, edge_state_dim),
            nn.SiLU(),
            nn.Linear(edge_state_dim, edge_state_dim),
        )

    def forward(self, scalars: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor, edge_state: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index
        if sender.numel() == 0:
            return edge_state
        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_state, edge_len.unsqueeze(-1)], dim=-1)
        return self.mlp(msg_in)

class NodeUpdateMLP(nn.Module):
    """Irrep-aware node update on scalars only (post-aggregation)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scalars, rest = h[..., : self.mlp[0].in_features], h[..., self.mlp[0].in_features :]
        scalars = scalars + self.mlp(scalars)
        return torch.cat([scalars, rest], dim=-1)

class FlashACE(nn.Module):
    def __init__(
        self,
        r_max=5.0,
        l_max=2,
        num_radial=8,
        hidden_dim=128,
        num_layers=2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        descriptor_passes: int = 1,
        descriptor_residual: bool = True,
        radial_mlp_hidden: int = 64,
        radial_mlp_layers: int = 2,
        attention_message_clip: float | None = None,
        attention_conditioned_decay: bool = True,
        attention_share_qkv: str | bool = "none",
        attention_scalar_pre_norm: bool = True,
        attention_layer_scale_init: float | None = 1e-2,
        drop_path_rate: float = 0.0,
        use_aux_force_head: bool = True,
        use_aux_stress_head: bool = True,
        message_passing_layers: int = 0,
        interleave_descriptor: bool = False,
        edge_update_per_layer: bool = False,
        edge_state_dim: int | None = None,
        node_update_mlp: bool = False,
        attention_edge_film: bool = False,
        attention_edge_film_hidden: int | None = None,
        attention_scalar_post_norm: bool = False,
        attention_scalar_post_gate: bool = False,
        attention_tensor_post_gate: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.l_max = l_max
        self.descriptor_passes = max(1, int(descriptor_passes))
        self.descriptor_residual = bool(descriptor_residual)
        self.attention_message_clip = attention_message_clip
        self.attention_conditioned_decay = attention_conditioned_decay
        self.attention_share_qkv = attention_share_qkv
        self.attention_scalar_pre_norm = attention_scalar_pre_norm
        self.attention_layer_scale_init = attention_layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.use_aux_force_head = use_aux_force_head
        self.use_aux_stress_head = use_aux_stress_head
        self.message_passing_layers = max(0, int(message_passing_layers))
        self.interleave_descriptor = bool(interleave_descriptor)
        self.edge_update_per_layer = bool(edge_update_per_layer)
        self.node_update_mlp = bool(node_update_mlp)
        self.edge_state_dim = int(edge_state_dim) if edge_state_dim is not None else hidden_dim
        self.edge_sh_irreps = o3.Irreps.spherical_harmonics(l_max)
        self.edge_sh_dim = self.edge_sh_irreps.dim
        self.edge_irreps = o3.Irreps(f"{hidden_dim}x0e + {max(1, hidden_dim//4)}x1e")
        self.node_scalar_irreps = o3.Irreps(f"{hidden_dim}x0e")
        self.attention_edge_film = bool(attention_edge_film)
        self.attention_edge_film_hidden = attention_edge_film_hidden
        self.attention_scalar_post_norm = bool(attention_scalar_post_norm)
        self.attention_scalar_post_gate = bool(attention_scalar_post_gate)
        self.attention_tensor_post_gate = bool(attention_tensor_post_gate)

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
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
        )

        self.mp_layers = nn.ModuleList(
            [ScalarMessagePassing(hidden_dim) for _ in range(self.message_passing_layers)]
        )
        self.edge_updates = nn.ModuleList(
            [EdgeUpdate(hidden_dim) for _ in range(num_layers)] if self.edge_update_per_layer else []
        )
        self.node_updates = nn.ModuleList(
            [NodeUpdateMLP(hidden_dim) for _ in range(num_layers)] if self.node_update_mlp else []
        )
        self.edge_tp_sender = o3.FullyConnectedTensorProduct(
            self.node_scalar_irreps, self.edge_sh_irreps, self.edge_irreps
        ) if self.edge_update_per_layer else None
        self.edge_tp_receiver = o3.FullyConnectedTensorProduct(
            self.node_scalar_irreps, self.edge_sh_irreps, self.edge_irreps
        ) if self.edge_update_per_layer else None

        dpr_values = torch.linspace(0, drop_path_rate, num_layers) if num_layers > 0 else torch.tensor([])
        self.layers = nn.ModuleList([
            DenseFlashAttention(
                self.ace.irreps_out,
                hidden_dim,
                message_clip=attention_message_clip,
                use_conditioned_decay=attention_conditioned_decay,
                share_qkv_mode=attention_share_qkv,
                scalar_pre_norm=attention_scalar_pre_norm,
                layer_scale_init_value=attention_layer_scale_init,
                drop_path_rate=float(dpr_values[i]) if len(dpr_values) > 0 else 0.0,
                edge_film=self.attention_edge_film,
                edge_film_hidden=self.attention_edge_film_hidden,
                scalar_post_norm=self.attention_scalar_post_norm,
                scalar_post_gate=self.attention_scalar_post_gate,
                tensor_post_gate=self.attention_tensor_post_gate,
            )
            for i in range(num_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.SiLU(), 
            nn.Linear(64, 1)
        )
        self.aux_force_head = None
        self.aux_stress_head = None
        if self.use_aux_force_head:
            self.aux_force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
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
        edge_len = torch.norm(edge_vec, dim=1)

        # 1. Descriptor iterations (optionally residual) before message passing / attention.
        h = self.emb(z)
        for i in range(self.descriptor_passes):
            scalars = h[..., : self.hidden_dim]
            desc = self.ace(scalars, edge_index, edge_vec, edge_len)
            if i == 0 or not self.descriptor_residual:
                h = desc
            else:
                h = h + desc

        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_len)

        edge_state = None
        edge_sh = None
        sender, receiver = edge_index
        if self.edge_update_per_layer:
            edge_sh = o3.spherical_harmonics(self.edge_sh_irreps, edge_vec, normalize=True, normalization="component")

        for idx, layer in enumerate(self.layers):
            if self.interleave_descriptor:
                scalars = h[..., : self.hidden_dim]
                desc = self.ace(scalars, edge_index, edge_vec, edge_len)
                h = h + desc if self.descriptor_residual else desc
            if self.edge_update_per_layer and len(self.edge_updates) > 0:
                h = self.edge_updates[idx](h, edge_index, edge_len)
            edge_attr = edge_state
            if self.edge_update_per_layer:
                h_scalars = h[..., : self.hidden_dim]
                edge_attr_sender = self.edge_tp_sender(h_scalars[sender], edge_sh)
                edge_attr_receiver = self.edge_tp_receiver(h_scalars[receiver], edge_sh)
                edge_state = edge_attr_sender + edge_attr_receiver
                edge_attr = edge_state
            h = layer(h, edge_index, edge_vec, edge_len, temperature_scale=temperature_scale, edge_attr=edge_attr)
            if self.node_update_mlp and len(self.node_updates) > 0:
                h = self.node_updates[idx](h)
            
        # 2. Readout
        # Note: We extract only the scalar (L=0) features for energy
        # The optimized physics.py puts scalars first, so this slice is correct.
        scalars = h[:, :self.hidden_dim] 
        E = torch.sum(self.readout(scalars))

        aux = {}
        if self.use_aux_force_head and self.aux_force_head is not None:
            aux['force'] = self.aux_force_head(scalars)
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
