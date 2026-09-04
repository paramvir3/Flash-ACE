from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from .physics import (
    ACE_Descriptor,
    ACEV2Descriptor,
    ACEV3TokenDescriptor,
    ACEV4CumulantTokenDescriptor,
    ACERadialBasis,
    SmoothACERadialBasis,
)


def _layer_scale_num_copies(blocks: tuple[tuple[int, int], ...]) -> int:
    """Number of irrep copies, i.e. the number of independent equivariant scales."""
    return int(sum(multiplicity for multiplicity, _ in blocks))


def _irrepwise_layer_scale(
    scale: torch.Tensor,
    blocks: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Broadcast one scalar per irrep copy to all of its magnetic components.

    Equivariance requires the residual multiplier to be constant over the
    ``2*l+1`` components ``m`` of a single irrep copy: only then does it commute
    with the Wigner rotation ``D^{(l)}``.  The parameter is therefore *stored*
    with one entry per copy, so a non-equivariant state is unrepresentable by
    construction rather than being projected away at forward time.

    ``scale`` has shape ``(sum_l mul_l,)`` and the return value has shape
    ``(sum_l mul_l * (2*l+1),)``, matching ``irreps.dim``.
    """
    pieces = []
    offset = 0
    for multiplicity, irrep_dim in blocks:
        copy_scale = scale[offset : offset + multiplicity]
        pieces.append(copy_scale.unsqueeze(-1).expand(multiplicity, irrep_dim).reshape(-1))
        offset += multiplicity
    return torch.cat(pieces)


def freeze_layer_scales(model: torch.nn.Module) -> int:
    """Bake the expanded residual layer scale as a constant, for inference only.

    ``_irrepwise_layer_scale`` broadcasts one scalar per irrep copy to all of its
    magnetic components. During inference the underlying parameter is fixed, so
    the broadcast is a compile-time constant that is otherwise recomputed on
    every step. Freezing it removes a ``(mul, 2l+1)`` expand-and-concatenate from
    the traced graph, which matters for ahead-of-time compilation, and saves the
    work at every MD step.

    Returns the number of blocks frozen. Call only on a model in ``eval()`` mode;
    training must keep the live parameter so gradients flow.
    """
    frozen = 0
    for module in model.modules():
        scale = getattr(module, "layer_scale_attn", None)
        blocks = getattr(module, "_layer_scale_blocks", None)
        if scale is None or blocks is None:
            continue
        module._frozen_layer_scale = _irrepwise_layer_scale(scale.detach(), blocks)
        frozen += 1
    return frozen


def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax over incoming local edges for each receiver atom."""
    if logits.numel() == 0:
        return logits

    logits_f = logits.float()
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per_node = torch.full(
        (num_nodes, logits.shape[-1]),
        float("-inf"),
        device=logits.device,
        dtype=logits_f.dtype,
    )
    max_per_node = max_per_node.scatter_reduce(
        0,
        expanded_index,
        logits_f,
        reduce="amax",
        include_self=True,
    )
    max_per_node = torch.where(
        torch.isfinite(max_per_node), max_per_node, torch.zeros_like(max_per_node)
    )
    centered = logits_f - max_per_node[index]
    exp_logits = torch.exp(centered)
    denom = torch.zeros_like(max_per_node).scatter_add(0, expanded_index, exp_logits)
    return (exp_logits / (denom[index] + 1e-9)).to(logits.dtype)


class ScalarPreNorm(nn.Module):
    """Normalize invariant scalar channels without mixing equivariant components."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scalars = self.norm(x[..., : self.hidden_dim])
        rest = x[..., self.hidden_dim :]
        return torch.cat((scalars, rest), dim=-1)


class LocalEquivariantAttentionBlock(nn.Module):
    """Local attention on ACE irreps with invariant weights and equivariant values.

    This is intentionally not a message-passing stack: the only communication is
    a single local transformer-style attention operation over cutoff neighbors.
    The attention logits are scalar invariants, while values are transformed with
    e3nn linear maps, so the residual update preserves E(3) equivariance.
    """

    def __init__(
        self,
        irreps,
        hidden_dim: int,
        r_max: float,
        num_radial: int,
        num_heads: int = 4,
        key_dim: int | None = None,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        use_distance_penalty: bool = True,
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.hidden_dim = hidden_dim
        self.num_heads = max(1, int(num_heads))
        self.key_dim = key_dim or max(16, hidden_dim // self.num_heads)
        self.use_distance_penalty = bool(use_distance_penalty)

        self.norm1 = ScalarPreNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
        self.radial_basis = ACERadialBasis(
            r_max,
            num_radial,
            envelope_exponent=envelope_exponent,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        self.radial_bias = nn.Sequential(
            nn.Linear(num_radial, max(16, self.num_heads * 4)),
            nn.SiLU(),
            nn.Linear(max(16, self.num_heads * 4), self.num_heads),
        )
        self.distance_log_scale = (
            nn.Parameter(torch.zeros(self.num_heads))
            if self.use_distance_penalty
            else None
        )

        self.value_proj = nn.ModuleList(
            [o3.Linear(self.irreps, self.irreps) for _ in range(self.num_heads)]
        )
        self.out_proj = o3.Linear(self.irreps, self.irreps)
        self.dropout = nn.Dropout(dropout)
        self._layer_scale_blocks = tuple(
            (int(multiplicity), int(irrep.dim)) for multiplicity, irrep in self.irreps
        )
        self.layer_scale_attn = (
            nn.Parameter(
                torch.full(
                    (_layer_scale_num_copies(self._layer_scale_blocks),),
                    float(layer_scale_init),
                )
            )
            if layer_scale_init is not None
            else None
        )

        ffn_hidden = ffn_hidden or hidden_dim * 4
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

    def _attention_layer_scale(self) -> torch.Tensor:
        """Expanded residual scale, using the frozen constant when available."""
        frozen = getattr(self, "_frozen_layer_scale", None)
        if frozen is not None:
            return frozen
        return _irrepwise_layer_scale(self.layer_scale_attn, self._layer_scale_blocks)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_len: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        if sender.numel() == 0:
            return x

        x_norm = self.norm1(x)
        scalars = x_norm[..., : self.hidden_dim]
        q = self.q_proj(scalars).view(-1, self.num_heads, self.key_dim)
        k = self.k_proj(scalars).view(-1, self.num_heads, self.key_dim)

        radial = self.radial_basis(edge_len)
        logits = (q[receiver] * k[sender]).sum(dim=-1) / math.sqrt(self.key_dim)
        logits = logits + self.radial_bias(radial)
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * edge_len[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)

        alpha = self.dropout(_segment_softmax(logits, receiver, x.shape[0]))
        out = torch.zeros_like(x)
        for head, value_layer in enumerate(self.value_proj):
            values = value_layer(x_norm)[sender]
            out.index_add_(0, receiver, alpha[:, head : head + 1].to(values.dtype) * values)
        out = out / self.num_heads
        out = self.out_proj(out)
        if self.layer_scale_attn is not None:
            out = out * self._attention_layer_scale()
        x = x + out

        scalars = x[..., : self.hidden_dim]
        rest = x[..., self.hidden_dim :]
        scalar_update = self.scalar_ffn(self.norm2(scalars))
        if self.layer_scale_ffn is not None:
            scalar_update = scalar_update * self.layer_scale_ffn
        return torch.cat((scalars + scalar_update, rest), dim=-1)

class LegacyTransformersACE(nn.Module):
    architecture_version = 1

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
        correlation_order: int = 4,
        correlation_channels: int = 16,
        attention_num_heads: int | None = None,
        attention_key_dim: int | None = None,
        attention_ffn_hidden: int | None = None,
        attention_dropout: float = 0.0,
        attention_layer_scale_init: float | None = 1e-2,
        attention_distance_penalty: bool = True,
        transformer_num_heads: int = 4,
        transformer_ffn_hidden: int | None = None,
        transformer_dropout: float = 0.0,
        use_aux_force_head: bool = True,
        use_aux_stress_head: bool = True,
        interleave_descriptor: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.l_max = l_max
        self.descriptor_passes = max(1, int(descriptor_passes))
        self.descriptor_residual = bool(descriptor_residual)
        self.attention_num_heads = max(1, int(attention_num_heads or transformer_num_heads))
        self.attention_key_dim = attention_key_dim
        self.attention_ffn_hidden = attention_ffn_hidden or transformer_ffn_hidden
        self.attention_dropout = float(attention_dropout if attention_dropout is not None else transformer_dropout)
        self.attention_layer_scale_init = attention_layer_scale_init
        self.attention_distance_penalty = bool(attention_distance_penalty)
        # Direct force/stress heads are intentionally disabled: MD forces and
        # stresses must come from derivatives of a single scalar energy.
        self.use_aux_force_head = False
        self.use_aux_stress_head = False
        self.interleave_descriptor = bool(interleave_descriptor)
        self.node_scalar_irreps = o3.Irreps(f"{hidden_dim}x0e")

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
        self.attention_irreps = self.ace.irreps_out

        self.layers = nn.ModuleList([
            LocalEquivariantAttentionBlock(
                self.attention_irreps,
                hidden_dim,
                r_max,
                num_radial,
                num_heads=self.attention_num_heads,
                key_dim=self.attention_key_dim,
                ffn_hidden=self.attention_ffn_hidden,
                dropout=self.attention_dropout,
                layer_scale_init=self.attention_layer_scale_init,
                radial_basis_type=radial_basis_type,
                radial_trainable=radial_trainable,
                envelope_exponent=envelope_exponent,
                gaussian_width=gaussian_width,
                use_distance_penalty=self.attention_distance_penalty,
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

    def forward(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
    ):
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        cell_volume = data.get('volume', None)
        cell = data.get('cell', None)
        edge_shift = data.get('edge_shift', None)

        if compute_stress is None:
            compute_stress = training

        # We always need gradients w.r.t. atomic positions to compute forces.
        # Detach to ensure we work with a leaf tensor before enabling grads.
        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and cell_volume is not None:
            # Parameterize the small-strain tensor symmetrically so the stress
            # we backpropagate through corresponds to the symmetric Cauchy
            # stress and does not pick up spurious rotational components. This
            # matches ACE-style stress evaluation by differentiating with
            # respect to symmetric lattice strains.
            strain_params = torch.zeros(6, device=pos.device, dtype=pos.dtype, requires_grad=True)

            epsilon = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]

            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            pos = pos @ deformation
            if cell is not None:
                cell = cell @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_shift = edge_shift.to(device=pos.device, dtype=pos.dtype)
            edge_vec = edge_vec + edge_shift @ cell
        edge_len = torch.norm(edge_vec, dim=1)

        # 1. Descriptor iterations (optionally residual) before local attention.
        h = self.emb(z)
        for i in range(self.descriptor_passes):
            scalars = h[..., : self.hidden_dim]
            desc = self.ace(scalars, edge_index, edge_vec, edge_len)
            if i == 0 or not self.descriptor_residual:
                h = desc
            else:
                h = h + desc

        for idx, layer in enumerate(self.layers):
            if self.interleave_descriptor:
                scalars = h[..., : self.hidden_dim]
                desc = self.ace(scalars, edge_index, edge_vec, edge_len)
                h = h + desc if self.descriptor_residual else desc
            h = layer(h, edge_index, edge_len, temperature_scale=temperature_scale)
            
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
        grads = torch.autograd.grad(
            E,
            pos,
            create_graph=training,  # only keep graph for higher-order grads when training
            retain_graph=training or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(pos)
        
        S = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            # Retain the graph so the outer loss.backward() can still traverse
            # the computation graph built when taking the strain derivative.
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=training,
                retain_graph=training,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                # Map the 6 unique components back to a symmetric stress tensor
                # and normalize by the deformed volume to avoid overestimating
                # stress under volumetric strain.
                stress = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
                stress[0, 0] = g_eps[0]
                stress[1, 1] = g_eps[1]
                stress[2, 2] = g_eps[2]
                # Each shear parameter changes two symmetric strain entries, so
                # its derivative is twice the corresponding tensor component.
                stress[0, 1] = stress[1, 0] = 0.5 * g_eps[3]
                stress[0, 2] = stress[2, 0] = 0.5 * g_eps[4]
                stress[1, 2] = stress[2, 1] = 0.5 * g_eps[5]

                if cell is not None:
                    volume = torch.det(cell).abs().clamp_min(1e-12)
                else:
                    volume = cell_volume * torch.det(deformation)
                # ASE uses sigma_ab = (1 / V) dE / d(epsilon_ab). Its cell
                # filters apply the minus sign when constructing cell forces.
                S = stress / volume

        return E, F, S, aux


def _segment_cutoff_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    cutoff: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Cutoff-weighted attention with a unit self/null channel per center.

    The null channel prevents normalization from cancelling the cutoff when all
    neighbors approach ``r_max``. Consequently the entire local update and its
    first two radial derivatives vanish with the C2 envelope.
    """
    if logits.numel() == 0:
        return logits

    logits_f = logits.float()
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per_node = torch.zeros(
        (num_nodes, logits.shape[-1]),
        device=logits.device,
        dtype=logits_f.dtype,
    )
    max_per_node = max_per_node.scatter_reduce(
        0,
        expanded_index,
        logits_f,
        reduce="amax",
        include_self=True,
    )
    numerator = cutoff.float()[:, None] * torch.exp(logits_f - max_per_node[index])
    null_weight = torch.exp(-max_per_node)
    denominator = null_weight.scatter_add(0, expanded_index, numerator)
    return (numerator / denominator[index].clamp_min(1e-12)).to(logits.dtype)


def _fixed_token_cutoff_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    cutoff: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Cutoff attention for v3's nonempty edge-plus-moment token set."""

    logits_f = logits.float()
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per_node = torch.zeros(
        (num_nodes, logits.shape[-1]),
        device=logits.device,
        dtype=logits_f.dtype,
    )
    max_per_node = max_per_node.scatter_reduce(
        0,
        expanded_index,
        logits_f,
        reduce="amax",
        include_self=True,
    )
    numerator = cutoff.float()[:, None] * torch.exp(logits_f - max_per_node[index])
    null_weight = torch.exp(-max_per_node)
    denominator = null_weight.scatter_add(0, expanded_index, numerator)
    return (numerator / denominator[index].clamp_min(1e-12)).to(logits.dtype)


class StrictLocalEquivariantAttentionBlock(nn.Module):
    """Neighbor-set attention without exchanging updated atomic hidden states."""

    def __init__(
        self,
        node_irreps,
        edge_irreps,
        hidden_dim: int,
        edge_scalar_dim: int,
        r_max: float,
        num_radial: int,
        num_heads: int = 2,
        key_dim: int | None = None,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        use_distance_penalty: bool = True,
        attention_mode: str = "full",
    ):
        super().__init__()
        # Ablation arms. Everything outside the logit construction is identical,
        # so an accuracy difference is attributable to the weighting itself and
        # not to a different aggregation, readout or training path.
        #
        #   full     l = (q.k)/sqrt(K) + b_h(r) - softplus(s_h) r   (as published)
        #   no_qk    l =                 b_h(r) - softplus(s_h) r   (distance only)
        #   uniform  l = 0            -> alpha is the cutoff-weighted mean
        #   none     the attention update is skipped entirely
        #
        # `no_qk` is the decisive control: it keeps a *learned radial filter*,
        # which is what plain ACE/MACE already provide, and removes only the
        # content-dependent query-key term. `uniform` drops learned weighting
        # too; `none` leaves the ACE descriptor plus the scalar FFN and readout.
        valid_modes = ("full", "no_qk", "uniform", "none")
        if attention_mode not in valid_modes:
            raise ValueError(
                f"attention_mode must be one of {valid_modes}, got {attention_mode!r}"
            )
        self.attention_mode = attention_mode
        self._uses_query_key = attention_mode == "full"
        self._uses_learned_logits = attention_mode in ("full", "no_qk")
        self._uses_attention_update = attention_mode != "none"
        self.node_irreps = o3.Irreps(node_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.hidden_dim = int(hidden_dim)
        self.edge_scalar_dim = int(edge_scalar_dim)
        self.num_heads = max(1, int(num_heads))
        self.key_dim = key_dim or max(8, hidden_dim // self.num_heads)

        self.node_norm = ScalarPreNorm(hidden_dim)
        # Projections an arm does not use are not created, so each arm reports
        # an honest parameter count rather than one padded with dead weights.
        if self._uses_query_key:
            self.edge_scalar_norm = nn.LayerNorm(self.edge_scalar_dim)
            self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
            self.k_proj = nn.Linear(
                self.edge_scalar_dim, self.num_heads * self.key_dim, bias=False
            )
        else:
            self.edge_scalar_norm = None
            self.q_proj = None
            self.k_proj = None
        self.radial_basis = SmoothACERadialBasis(
            r_max,
            num_radial,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        radial_hidden = max(16, self.num_heads * 4)
        self.radial_bias = (
            nn.Sequential(
                nn.Linear(num_radial, radial_hidden),
                nn.SiLU(),
                nn.Linear(radial_hidden, self.num_heads),
            )
            if self._uses_learned_logits
            else None
        )
        self.distance_log_scale = (
            nn.Parameter(torch.zeros(self.num_heads))
            if use_distance_penalty and self._uses_learned_logits
            else None
        )

        # The `none` arm performs no edge aggregation, so the equivariant value
        # and output projections would be dead weight; leaving them out keeps its
        # reported parameter count equal to what it actually uses.
        self.value_proj = (
            nn.ModuleList(
                [o3.Linear(self.edge_irreps, self.node_irreps) for _ in range(self.num_heads)]
            )
            if self._uses_attention_update
            else None
        )
        self.out_proj = (
            o3.Linear(self.node_irreps, self.node_irreps)
            if self._uses_attention_update
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self._layer_scale_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.node_irreps
        )
        self.layer_scale_attn = (
            nn.Parameter(
                torch.full(
                    (_layer_scale_num_copies(self._layer_scale_blocks),),
                    float(layer_scale_init),
                )
            )
            if layer_scale_init is not None and self._uses_attention_update
            else None
        )

        self.non_scalar_irreps = o3.Irreps(self.node_irreps[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        invariant_dim = hidden_dim + self.non_scalar_norm.irreps_out.dim
        ffn_hidden = ffn_hidden or hidden_dim * 2
        self.scalar_norm = nn.LayerNorm(invariant_dim)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(invariant_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

    def _apply_scalar_ffn(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[:, : self.hidden_dim]
        rest = x[:, self.hidden_dim :]
        non_scalar_invariants = self.non_scalar_norm(rest)
        invariants = torch.cat((scalars, non_scalar_invariants), dim=-1)
        scalar_update = self.scalar_ffn(self.scalar_norm(invariants))
        if self.layer_scale_ffn is not None:
            scalar_update = scalar_update * self.layer_scale_ffn
        return torch.cat((scalars + scalar_update, rest), dim=-1)

    def _attention_layer_scale(self) -> torch.Tensor:
        """Expanded residual scale, using the frozen constant when available."""
        frozen = getattr(self, "_frozen_layer_scale", None)
        if frozen is not None:
            return frozen
        return _irrepwise_layer_scale(self.layer_scale_attn, self._layer_scale_blocks)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: torch.Tensor,
        receiver: torch.Tensor,
        edge_len: torch.Tensor,
        cutoff: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        if receiver.numel() == 0 or not self._uses_attention_update:
            # `none` still applies the scalar FFN, so the arms differ only by the
            # attention update and not by the presence of a nonlinearity.
            return self._apply_scalar_ffn(x)

        if self._uses_query_key:
            x_norm = self.node_norm(x)
            queries = self.q_proj(x_norm[:, : self.hidden_dim]).view(
                -1, self.num_heads, self.key_dim
            )
            edge_scalars = self.edge_scalar_norm(
                edge_features[:, : self.edge_scalar_dim]
            )
            keys = self.k_proj(edge_scalars).view(-1, self.num_heads, self.key_dim)
            logits = (queries[receiver] * keys).sum(dim=-1) / math.sqrt(self.key_dim)
        else:
            # Same shape and dtype as the full arm, so the softmax, the cutoff
            # weighting and the aggregation below are bit-for-bit the same code.
            logits = torch.zeros(
                (receiver.shape[0], self.num_heads),
                dtype=edge_features.dtype,
                device=edge_features.device,
            )
        if self.radial_bias is not None:
            logits = logits + self.radial_bias(self.radial_basis(edge_len))
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * edge_len[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)

        alpha = _segment_cutoff_softmax(logits, receiver, cutoff, x.shape[0])
        alpha = self.dropout(alpha)
        update = torch.zeros_like(x)
        for head, value_layer in enumerate(self.value_proj):
            values = value_layer(edge_features)
            update.index_add_(
                0,
                receiver,
                alpha[:, head : head + 1].to(values.dtype) * values,
            )
        update = self.out_proj(update / self.num_heads)
        if self.layer_scale_attn is not None:
            update = update * self._attention_layer_scale()
        x = x + update

        return self._apply_scalar_ffn(x)


class TransformersACE(nn.Module):
    """Corrected strictly local ACE-attention potential (architecture v2)."""

    architecture_version = 2

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
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        attention_num_heads: int | None = None,
        attention_key_dim: int | None = None,
        attention_ffn_hidden: int | None = None,
        attention_dropout: float = 0.0,
        attention_layer_scale_init: float | None = 1e-2,
        attention_distance_penalty: bool = True,
        attention_mode: str = "full",
        transformer_num_heads: int = 4,
        transformer_ffn_hidden: int | None = None,
        transformer_dropout: float = 0.0,
        use_aux_force_head: bool = False,
        use_aux_stress_head: bool = False,
        interleave_descriptor: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.r_max = float(r_max)
        self.l_max = int(l_max)
        self.descriptor_passes = 1
        self.descriptor_residual = True
        self.interleave_descriptor = False
        self.use_aux_force_head = False
        self.use_aux_stress_head = False

        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACEV2Descriptor(
            r_max=r_max,
            l_max=l_max,
            num_radial=num_radial,
            hidden_dim=hidden_dim,
            correlation_order=correlation_order,
            correlation_channels=correlation_channels,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
        )
        self.attention_irreps = self.ace.irreps_out
        attention_heads = max(1, int(attention_num_heads or transformer_num_heads))
        attention_hidden = attention_ffn_hidden or transformer_ffn_hidden
        attention_dropout = float(
            attention_dropout if attention_dropout is not None else transformer_dropout
        )
        edge_scalar_dim = self.ace.irreps_correlation[0].mul
        self.attention_mode = str(attention_mode)
        self.layers = nn.ModuleList(
            [
                StrictLocalEquivariantAttentionBlock(
                    node_irreps=self.attention_irreps,
                    edge_irreps=self.ace.irreps_correlation,
                    hidden_dim=hidden_dim,
                    edge_scalar_dim=edge_scalar_dim,
                    r_max=r_max,
                    num_radial=num_radial,
                    num_heads=attention_heads,
                    key_dim=attention_key_dim,
                    ffn_hidden=attention_hidden,
                    dropout=attention_dropout,
                    layer_scale_init=attention_layer_scale_init,
                    radial_basis_type=radial_basis_type,
                    radial_trainable=radial_trainable,
                    gaussian_width=gaussian_width,
                    use_distance_penalty=attention_distance_penalty,
                    attention_mode=attention_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.aux_force_head = None
        self.aux_stress_head = None

    def forward_batched(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
    ):
        """Evaluate several structures as one disconnected graph.

        ``data['batch']`` maps each atom to its structure, ``data['cell']`` and
        ``data['volume']`` carry a leading structure dimension, and the edge
        indices are already offset into the concatenated atom list.  Returns
        *per-structure* energies and stresses; forces stay per atom.

        This exists because the GPU is launch-bound for single small cells: one
        batched graph issues the same number of kernels as one structure but
        does B times the work inside them.  The single-structure ``forward`` is
        left byte-identical so the deployed TorchScript/AOT paths cannot regress.
        """
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        batch = data['batch']
        n_struct = int(data['n_structures'])
        cell = data.get('cell', None)
        edge_shift = data.get('edge_shift', None)
        cell_volume = data.get('volume', None)

        if compute_stress is None:
            compute_stress = training

        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and (cell is not None or cell_volume is not None):
            strain_params = torch.zeros(
                n_struct, 6, device=pos.device, dtype=pos.dtype, requires_grad=True
            )
            epsilon = torch.zeros(n_struct, 3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[:, 0, 0] = strain_params[:, 0]
            epsilon[:, 1, 1] = strain_params[:, 1]
            epsilon[:, 2, 2] = strain_params[:, 2]
            epsilon[:, 0, 1] = epsilon[:, 1, 0] = strain_params[:, 3]
            epsilon[:, 0, 2] = epsilon[:, 2, 0] = strain_params[:, 4]
            epsilon[:, 1, 2] = epsilon[:, 2, 1] = strain_params[:, 5]
            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype).unsqueeze(0) + epsilon
            pos = torch.bmm(pos.unsqueeze(1), deformation[batch]).squeeze(1)
            if cell is not None:
                cell = torch.bmm(cell, deformation)
        else:
            strain_params = None
            epsilon = None
            deformation = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_shift = edge_shift.to(device=pos.device, dtype=pos.dtype)
            edge_cell = cell[batch[edge_index[1]]]
            edge_vec = edge_vec + torch.bmm(edge_shift.unsqueeze(1), edge_cell).squeeze(1)
        edge_len = torch.norm(edge_vec, dim=1)

        h, edge_features, cutoff = self.ace(
            self.emb(z), edge_index, edge_vec, edge_len, return_edge_features=True
        )
        receiver = edge_index[1]
        for layer in self.layers:
            h = layer(
                h, edge_features, receiver, edge_len, cutoff,
                temperature_scale=temperature_scale,
            )

        atom_energy = self.readout(h[:, : self.hidden_dim]).squeeze(-1)
        E = torch.zeros(n_struct, device=pos.device, dtype=atom_energy.dtype)
        E = E.index_add(0, batch, atom_energy)
        total = E.sum()

        grads = torch.autograd.grad(
            total, pos,
            create_graph=training,
            retain_graph=training or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(pos)

        S = torch.zeros(n_struct, 3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            g_eps = torch.autograd.grad(
                total, strain_params,
                create_graph=training, retain_graph=training, allow_unused=True,
            )[0]
            if g_eps is not None:
                stress = torch.zeros(n_struct, 3, 3, device=pos.device, dtype=pos.dtype)
                stress[:, 0, 0] = g_eps[:, 0]
                stress[:, 1, 1] = g_eps[:, 1]
                stress[:, 2, 2] = g_eps[:, 2]
                stress[:, 0, 1] = stress[:, 1, 0] = 0.5 * g_eps[:, 3]
                stress[:, 0, 2] = stress[:, 2, 0] = 0.5 * g_eps[:, 4]
                stress[:, 1, 2] = stress[:, 2, 1] = 0.5 * g_eps[:, 5]
                if cell is not None:
                    volume = torch.det(cell).abs().clamp_min(1e-12)
                else:
                    volume = cell_volume * torch.det(deformation)
                S = stress / volume.view(-1, 1, 1)

        return E, F, S, {}

    def forward(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
    ):
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        cell_volume = data.get('volume', None)
        cell = data.get('cell', None)
        edge_shift = data.get('edge_shift', None)

        if compute_stress is None:
            compute_stress = training

        # Keep this derivative and strain path identical to architecture v1.
        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and cell_volume is not None:
            strain_params = torch.zeros(6, device=pos.device, dtype=pos.dtype, requires_grad=True)

            epsilon = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]

            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            pos = pos @ deformation
            if cell is not None:
                cell = cell @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_shift = edge_shift.to(device=pos.device, dtype=pos.dtype)
            edge_vec = edge_vec + edge_shift @ cell
        edge_len = torch.norm(edge_vec, dim=1)

        h, edge_features, cutoff = self.ace(
            self.emb(z),
            edge_index,
            edge_vec,
            edge_len,
            return_edge_features=True,
        )
        receiver = edge_index[1]
        for layer in self.layers:
            h = layer(
                h,
                edge_features,
                receiver,
                edge_len,
                cutoff,
                temperature_scale=temperature_scale,
            )

        scalars = h[:, : self.hidden_dim]
        E = torch.sum(self.readout(scalars))
        aux = {}

        grads = torch.autograd.grad(
            E,
            pos,
            create_graph=training,
            retain_graph=training or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(pos)

        S = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=training,
                retain_graph=training,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                stress = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
                stress[0, 0] = g_eps[0]
                stress[1, 1] = g_eps[1]
                stress[2, 2] = g_eps[2]
                stress[0, 1] = stress[1, 0] = 0.5 * g_eps[3]
                stress[0, 2] = stress[2, 0] = 0.5 * g_eps[4]
                stress[1, 2] = stress[2, 1] = 0.5 * g_eps[5]

                if cell is not None:
                    volume = torch.det(cell).abs().clamp_min(1e-12)
                else:
                    volume = cell_volume * torch.det(deformation)
                S = stress / volume

        return E, F, S, aux


class TensorialFixedEnvironmentAttentionBlock(nn.Module):
    """Equivariant cross-attention from one center to frozen ACE tokens.

    A token may describe a directed neighbor density or a body-ordered ACE
    moment.  Keys and values are functions only of those fixed tokens.  The
    only evolving state is the receiving center feature, so stacking blocks
    cannot expand the physical receptive field.
    """

    def __init__(
        self,
        node_irreps,
        token_irreps,
        hidden_dim: int,
        token_scalar_dim: int,
        r_max: float,
        num_radial: int,
        num_token_kinds: int,
        num_heads: int = 2,
        key_dim: int | None = None,
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        use_distance_penalty: bool = True,
    ):
        super().__init__()
        self.node_irreps = o3.Irreps(node_irreps)
        self.token_irreps = o3.Irreps(token_irreps)
        self.score_irreps = self.token_irreps
        self.hidden_dim = int(hidden_dim)
        self.token_scalar_dim = int(token_scalar_dim)
        self.num_heads = max(1, int(num_heads))
        self.key_dim = key_dim or max(8, hidden_dim // self.num_heads)

        self.node_norm = ScalarPreNorm(hidden_dim)
        self.token_scalar_norm = nn.LayerNorm(self.token_scalar_dim)
        self.token_kind_embedding = nn.Embedding(num_token_kinds, self.token_scalar_dim)
        self.q_scalar = nn.Linear(hidden_dim, self.num_heads * self.key_dim, bias=False)
        self.k_scalar = nn.Linear(
            self.token_scalar_dim,
            self.num_heads * self.key_dim,
            bias=False,
        )

        # Matching irreps are contracted to 0e scalars.  In the real e3nn
        # basis the normalized inner product is the corresponding CG scalar.
        self.q_tensor = o3.Linear(self.node_irreps, self.score_irreps)
        self.k_tensor = o3.Linear(self.token_irreps, self.score_irreps)
        self.score_channels = sum(mul for mul, _ in self.score_irreps)
        # Keep only primitive static metadata in the forward path.  e3nn's
        # Irrep objects intentionally do not implement ``len``; iterating over
        # them during forward prevents torch.export/AOTInductor capture.
        self._score_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.score_irreps
        )
        self.multipole_score = nn.Linear(self.score_channels, self.num_heads, bias=False)

        self.radial_basis = SmoothACERadialBasis(
            r_max,
            num_radial,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        radial_hidden = max(16, self.num_heads * 4)
        self.radial_bias = nn.Sequential(
            nn.Linear(num_radial, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, self.num_heads),
        )
        self.distance_log_scale = (
            nn.Parameter(torch.zeros(self.num_heads))
            if use_distance_penalty
            else None
        )

        self.value_tp = nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    self.node_irreps,
                    self.token_irreps,
                    self.node_irreps,
                    internal_weights=True,
                    shared_weights=True,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.out_proj = o3.Linear(self.node_irreps, self.node_irreps)
        self.dropout = nn.Dropout(dropout)
        self._layer_scale_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.node_irreps
        )
        self.layer_scale_attn = (
            nn.Parameter(
                torch.full(
                    (_layer_scale_num_copies(self._layer_scale_blocks),),
                    float(layer_scale_init),
                )
            )
            if layer_scale_init is not None
            else None
        )

        self.non_scalar_irreps = o3.Irreps(self.node_irreps[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        invariant_dim = hidden_dim + self.non_scalar_norm.irreps_out.dim
        ffn_hidden = ffn_hidden or hidden_dim * 2
        self.scalar_norm = nn.LayerNorm(invariant_dim)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(invariant_dim, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

    def _multipole_invariants(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        receiver: torch.Tensor,
    ) -> torch.Tensor:
        terms = []
        offset = 0
        for multiplicity, irrep_dim in self._score_blocks:
            width = multiplicity * irrep_dim
            q_block = query[:, offset : offset + width].reshape(
                query.shape[0], multiplicity, irrep_dim
            )
            k_block = key[:, offset : offset + width].reshape(
                key.shape[0], multiplicity, irrep_dim
            )
            terms.append(
                (q_block[receiver] * k_block).sum(dim=-1) / math.sqrt(irrep_dim)
            )
            offset += width
        return torch.cat(terms, dim=-1)

    def _apply_scalar_ffn(self, x: torch.Tensor) -> torch.Tensor:
        scalars = x[:, : self.hidden_dim]
        rest = x[:, self.hidden_dim :]
        invariants = torch.cat((scalars, self.non_scalar_norm(rest)), dim=-1)
        update = self.scalar_ffn(self.scalar_norm(invariants))
        if self.layer_scale_ffn is not None:
            update = update * self.layer_scale_ffn
        return torch.cat((scalars + update, rest), dim=-1)

    def _attention_layer_scale(self) -> torch.Tensor:
        """Expanded residual scale, using the frozen constant when available."""
        frozen = getattr(self, "_frozen_layer_scale", None)
        if frozen is not None:
            return frozen
        return _irrepwise_layer_scale(self.layer_scale_attn, self._layer_scale_blocks)

    def forward(
        self,
        x: torch.Tensor,
        token_features: torch.Tensor,
        token_receiver: torch.Tensor,
        token_length: torch.Tensor,
        token_cutoff: torch.Tensor,
        token_kind: torch.Tensor,
        temperature_scale: float = 1.0,
    ) -> torch.Tensor:
        x_norm = self.node_norm(x)
        q_scalar = self.q_scalar(x_norm[:, : self.hidden_dim]).view(
            -1, self.num_heads, self.key_dim
        )
        token_scalars = self.token_scalar_norm(
            token_features[:, : self.token_scalar_dim]
        ) + self.token_kind_embedding(token_kind)
        k_scalar = self.k_scalar(token_scalars).view(
            -1, self.num_heads, self.key_dim
        )
        logits = (q_scalar[token_receiver] * k_scalar).sum(dim=-1) / math.sqrt(
            self.key_dim
        )

        q_tensor = self.q_tensor(x_norm)
        k_tensor = self.k_tensor(token_features)
        tensor_scores = self._multipole_invariants(q_tensor, k_tensor, token_receiver)
        logits = logits + self.multipole_score(tensor_scores)
        logits = logits + self.radial_bias(self.radial_basis(token_length))
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * token_length[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)

        alpha = self.dropout(
            _fixed_token_cutoff_softmax(logits, token_receiver, token_cutoff, x.shape[0])
        )
        update = torch.zeros_like(x)
        for head, value_layer in enumerate(self.value_tp):
            values = value_layer(x_norm[token_receiver], token_features)
            update.index_add_(
                0,
                token_receiver,
                alpha[:, head : head + 1].to(values.dtype) * values,
            )
        update = self.out_proj(update / self.num_heads)
        if self.layer_scale_attn is not None:
            update = update * self._attention_layer_scale()
        return self._apply_scalar_ffn(x + update)


class CumulantMultiQueryTensorialAttentionBlock(TensorialFixedEnvironmentAttentionBlock):
    """v4 local cross-attention with shared tensor values and invariant gates.

    Heads retain independent scalar and multipole queries/keys, while a single
    equivariant tensor product supplies the token value.  This removes the
    v3 ``num_heads``-fold tensor-product cost without introducing a sender
    hidden-state dependency.  Additive smooth shell logits are equivalent to a
    multiplicative shell factor before the per-center softmax.
    """

    def __init__(self, *args, num_shells: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_shells = max(1, int(num_shells))
        self.shared_value_tp = o3.FullyConnectedTensorProduct(
            self.node_irreps, self.token_irreps, self.node_irreps,
            internal_weights=True, shared_weights=True,
        )
        del self.value_tp
        self.value_head_gate = nn.Linear(self.hidden_dim, self.num_heads)
        centers = torch.linspace(0.0, self.radial_basis.cutoff.r_max, self.num_shells)
        self.register_buffer("shell_centers", centers)
        self.shell_log_width = nn.Parameter(torch.zeros(self.num_shells))
        self.shell_bias = nn.Linear(self.num_shells, self.num_heads, bias=False)

        self._non_scalar_blocks = tuple(
            (int(multiplicity), int(irrep.dim))
            for multiplicity, irrep in self.non_scalar_irreps
        )
        self._non_scalar_multiplicities = sum(multiplicity for multiplicity, _ in self._non_scalar_blocks)
        if self._non_scalar_multiplicities:
            invariant_dim = int(self.scalar_norm.normalized_shape[0])
            self.tensor_swiglu = nn.Sequential(
                nn.LayerNorm(invariant_dim),
                nn.Linear(invariant_dim, 2 * self._non_scalar_multiplicities),
            )
            # Identity initialization provides a conservative starting point.
            nn.init.zeros_(self.tensor_swiglu[-1].weight)
            nn.init.zeros_(self.tensor_swiglu[-1].bias)
        else:
            self.tensor_swiglu = None

    def _shell_features(self, token_length: torch.Tensor, token_cutoff: torch.Tensor) -> torch.Tensor:
        width = F.softplus(self.shell_log_width).clamp_min(1e-4)
        distances = token_length[:, None] - self.shell_centers[None, :]
        return torch.exp(-0.5 * (distances / width[None, :]).square()) * token_cutoff[:, None]

    def _apply_scalar_ffn(self, x: torch.Tensor) -> torch.Tensor:
        scalars, rest = x[:, : self.hidden_dim], x[:, self.hidden_dim :]
        invariants = torch.cat((scalars, self.non_scalar_norm(rest)), dim=-1)
        scalar_update = self.scalar_ffn(self.scalar_norm(invariants))
        if self.layer_scale_ffn is not None:
            scalar_update = scalar_update * self.layer_scale_ffn
        if self.tensor_swiglu is None:
            return torch.cat((scalars + scalar_update, rest), dim=-1)

        first, second = self.tensor_swiglu(invariants).chunk(2, dim=-1)
        gates = 0.1 * torch.tanh(F.silu(first) * second)
        blocks, feature_offset, gate_offset = [], 0, 0
        for multiplicity, irrep_dim in self._non_scalar_blocks:
            width = multiplicity * irrep_dim
            block = rest[:, feature_offset : feature_offset + width].reshape(-1, multiplicity, irrep_dim)
            gate = gates[:, gate_offset : gate_offset + multiplicity].unsqueeze(-1)
            blocks.append((block * (1.0 + gate)).reshape(-1, width))
            feature_offset += width
            gate_offset += multiplicity
        return torch.cat((scalars + scalar_update, *blocks), dim=-1)

    def forward(
        self, x, token_features, token_receiver, token_length, token_cutoff,
        token_kind, temperature_scale: float = 1.0,
    ):
        x_norm = self.node_norm(x)
        q_scalar = self.q_scalar(x_norm[:, : self.hidden_dim]).view(-1, self.num_heads, self.key_dim)
        token_scalars = self.token_scalar_norm(token_features[:, : self.token_scalar_dim])
        token_scalars = token_scalars + self.token_kind_embedding(token_kind)
        k_scalar = self.k_scalar(token_scalars).view(-1, self.num_heads, self.key_dim)
        logits = (q_scalar[token_receiver] * k_scalar).sum(dim=-1) / math.sqrt(self.key_dim)
        q_tensor, k_tensor = self.q_tensor(x_norm), self.k_tensor(token_features)
        logits = logits + self.multipole_score(self._multipole_invariants(q_tensor, k_tensor, token_receiver))
        logits = logits + self.radial_bias(self.radial_basis(token_length))
        logits = logits + self.shell_bias(self._shell_features(token_length, token_cutoff))
        if self.distance_log_scale is not None:
            logits = logits - F.softplus(self.distance_log_scale)[None, :] * token_length[:, None]
        logits = logits / max(float(temperature_scale), 1e-4)
        alpha = self.dropout(_fixed_token_cutoff_softmax(logits, token_receiver, token_cutoff, x.shape[0]))

        # The coefficient is scalar and center-local; only the fixed token
        # enters the shared equivariant value tensor product.
        values = self.shared_value_tp(x_norm[token_receiver], token_features)
        head_gate = torch.sigmoid(self.value_head_gate(x_norm[:, : self.hidden_dim]))
        coefficient = (alpha * head_gate[token_receiver]).mean(dim=-1, keepdim=True)
        update = torch.zeros_like(x)
        update.index_add_(0, token_receiver, coefficient.to(values.dtype) * values)
        update = self.out_proj(update)
        if self.layer_scale_attn is not None:
            update = update * self._attention_layer_scale()
        return self._apply_scalar_ffn(x + update)


class InvariantIrrepReadout(nn.Module):
    """Energy readout from scalar channels and explicit tensor norms."""

    def __init__(self, irreps, hidden_dim: int):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.hidden_dim = int(hidden_dim)
        self.non_scalar_irreps = o3.Irreps(self.irreps[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        input_dim = hidden_dim + self.non_scalar_norm.irreps_out.dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        invariants = torch.cat(
            (x[:, : self.hidden_dim], self.non_scalar_norm(x[:, self.hidden_dim :])),
            dim=-1,
        )
        return self.net(invariants)


class TransformersACEV3(nn.Module):
    """TRACE v3: fixed-environment tensorial cross-attention potential."""

    architecture_version = 3

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
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        attention_num_heads: int | None = None,
        attention_key_dim: int | None = None,
        attention_ffn_hidden: int | None = None,
        attention_dropout: float = 0.0,
        attention_layer_scale_init: float | None = 1e-2,
        attention_distance_penalty: bool = True,
        transformer_num_heads: int = 4,
        transformer_ffn_hidden: int | None = None,
        transformer_dropout: float = 0.0,
        use_aux_force_head: bool = False,
        use_aux_stress_head: bool = False,
        interleave_descriptor: bool = False,
    ):
        super().__init__()
        if descriptor_passes != 1 or not descriptor_residual or interleave_descriptor:
            raise ValueError(
                "TRACE v3 uses one fixed ACE environment encoder; descriptor iteration "
                "and interleaving are not valid v3 options"
            )
        if envelope_exponent != 5:
            raise ValueError("TRACE v3 uses a fixed quintic C2 cutoff (envelope_exponent=5)")
        if use_aux_force_head or use_aux_stress_head:
            raise ValueError("TRACE v3 uses energy-derived forces and stress only")

        self.hidden_dim = int(hidden_dim)
        self.r_max = float(r_max)
        self.l_max = int(l_max)
        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACEV3TokenDescriptor(
            r_max=r_max,
            l_max=l_max,
            num_radial=num_radial,
            hidden_dim=hidden_dim,
            correlation_order=correlation_order,
            correlation_channels=correlation_channels,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
        )
        self.attention_irreps = self.ace.irreps_out
        attention_heads = max(1, int(attention_num_heads or transformer_num_heads))
        attention_hidden = attention_ffn_hidden or transformer_ffn_hidden
        if attention_dropout == 0.0 and transformer_dropout != 0.0:
            attention_dropout = transformer_dropout
        token_scalar_dim = self.ace.irreps_correlation[0].mul
        self.layers = nn.ModuleList(
            [
                TensorialFixedEnvironmentAttentionBlock(
                    node_irreps=self.attention_irreps,
                    token_irreps=self.ace.irreps_correlation,
                    hidden_dim=hidden_dim,
                    token_scalar_dim=token_scalar_dim,
                    r_max=r_max,
                    num_radial=num_radial,
                    num_token_kinds=self.ace.num_moment_tokens + 1,
                    num_heads=attention_heads,
                    key_dim=attention_key_dim,
                    ffn_hidden=attention_hidden,
                    dropout=float(attention_dropout),
                    layer_scale_init=attention_layer_scale_init,
                    radial_basis_type=radial_basis_type,
                    radial_trainable=radial_trainable,
                    gaussian_width=gaussian_width,
                    use_distance_penalty=attention_distance_penalty,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = InvariantIrrepReadout(self.attention_irreps, hidden_dim)

    def forward(
        self,
        data,
        training=False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
    ):
        z, pos, edge_index = data["z"], data["pos"], data["edge_index"]
        cell_volume = data.get("volume", None)
        cell = data.get("cell", None)
        edge_shift = data.get("edge_shift", None)
        if compute_stress is None:
            compute_stress = training

        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)
        if cell is not None:
            cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and cell_volume is not None:
            strain_params = torch.zeros(6, device=pos.device, dtype=pos.dtype, requires_grad=True)
            epsilon = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]
            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            pos = pos @ deformation
            if cell is not None:
                cell = cell @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift is not None and cell is not None:
            edge_vec = edge_vec + edge_shift.to(device=pos.device, dtype=pos.dtype) @ cell
        edge_len = torch.norm(edge_vec, dim=1)
        h, tokens, receiver, token_length, token_cutoff, token_kind = self.ace(
            self.emb(z), edge_index, edge_vec, edge_len
        )
        for layer in self.layers:
            h = layer(
                h,
                tokens,
                receiver,
                token_length,
                token_cutoff,
                token_kind,
                temperature_scale=temperature_scale,
            )
        E = torch.sum(self.readout(h))

        grads = torch.autograd.grad(
            E,
            pos,
            create_graph=training,
            retain_graph=training or epsilon is not None,
            allow_unused=True,
        )[0]
        F = -grads if grads is not None else torch.zeros_like(pos)

        S = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
        if compute_stress and epsilon is not None:
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=training,
                retain_graph=training,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                stress = torch.zeros(3, 3, device=pos.device, dtype=pos.dtype)
                stress[0, 0], stress[1, 1], stress[2, 2] = g_eps[:3]
                stress[0, 1] = stress[1, 0] = 0.5 * g_eps[3]
                stress[0, 2] = stress[2, 0] = 0.5 * g_eps[4]
                stress[1, 2] = stress[2, 1] = 0.5 * g_eps[5]
                volume = (
                    torch.det(cell).abs().clamp_min(1e-12)
                    if cell is not None
                    else cell_volume * torch.det(deformation)
                )
                S = stress / volume
        return E, F, S, {}


class TransformersACEV4(TransformersACEV3):
    """TRACE v4: connected local ACE tokens and shared-value cross-attention.

    TRACE v4 preserves v3's finite receptive field: each atom attends only to
    fixed descriptors of its original local environment.  It deliberately does
    not exchange hidden states between atoms and is therefore not a graph
    message-passing architecture.
    """

    architecture_version = 4

    def __init__(
        self,
        *args,
        attention_num_shells: int = 4,
        correlation_rank_initial: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ace = ACEV4CumulantTokenDescriptor(
            r_max=self.r_max,
            l_max=self.l_max,
            num_radial=kwargs.get("num_radial", 8),
            hidden_dim=self.hidden_dim,
            correlation_order=kwargs.get("correlation_order", 4),
            correlation_channels=kwargs.get("correlation_channels", 16),
            radial_basis_type=kwargs.get("radial_basis_type", "bessel"),
            radial_trainable=kwargs.get("radial_trainable", False),
            gaussian_width=kwargs.get("gaussian_width", 0.5),
            radial_mlp_hidden=kwargs.get("radial_mlp_hidden", 32),
            radial_mlp_layers=kwargs.get("radial_mlp_layers", 2),
        )
        self.attention_irreps = self.ace.irreps_out
        attention_heads = max(1, int(kwargs.get("attention_num_heads") or kwargs.get("transformer_num_heads", 4)))
        attention_hidden = kwargs.get("attention_ffn_hidden") or kwargs.get("transformer_ffn_hidden")
        attention_dropout = kwargs.get("attention_dropout", 0.0)
        if attention_dropout == 0.0 and kwargs.get("transformer_dropout", 0.0) != 0.0:
            attention_dropout = kwargs["transformer_dropout"]
        self.layers = nn.ModuleList(
            [
                CumulantMultiQueryTensorialAttentionBlock(
                    node_irreps=self.attention_irreps,
                    token_irreps=self.ace.irreps_correlation,
                    hidden_dim=self.hidden_dim,
                    token_scalar_dim=self.ace.irreps_correlation[0].mul,
                    r_max=self.r_max,
                    num_radial=kwargs.get("num_radial", 8),
                    num_token_kinds=self.ace.num_moment_tokens + 1,
                    num_heads=attention_heads,
                    key_dim=kwargs.get("attention_key_dim"),
                    ffn_hidden=attention_hidden,
                    dropout=float(attention_dropout),
                    layer_scale_init=kwargs.get("attention_layer_scale_init", 1e-2),
                    radial_basis_type=kwargs.get("radial_basis_type", "bessel"),
                    radial_trainable=kwargs.get("radial_trainable", False),
                    gaussian_width=kwargs.get("gaussian_width", 0.5),
                    use_distance_penalty=kwargs.get("attention_distance_penalty", True),
                    num_shells=attention_num_shells,
                )
                for _ in range(len(self.layers))
            ]
        )
        self.readout = InvariantIrrepReadout(self.attention_irreps, self.hidden_dim)
        if correlation_rank_initial is not None:
            self.ace.set_correlation_rank(correlation_rank_initial)

    def set_correlation_rank(self, rank: int) -> None:
        self.ace.set_correlation_rank(rank)


# The historical project name now points to v2 for newly created models. The
# calculator selects LegacyTransformersACE for checkpoints without a version.
FlashACE = TransformersACE
