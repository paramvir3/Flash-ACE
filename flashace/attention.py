import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

_torch_scatter_available = importlib.util.find_spec("torch_scatter") is not None
if _torch_scatter_available:
    from torch_scatter import scatter_add, scatter_softmax


class LocalMessagePassing(nn.Module):
    """Lightweight locality-preserving block to sharpen force learning.

    This module aggregates neighbor features with a distance-decaying scalar
    kernel, then mixes them back into the node representation with an equivariant
    linear map. Because the weights depend only on distances (scalars), the
    operation preserves the irreps structure while reinforcing short-range
    correlations.
    """

    def __init__(self, irreps_in, sharpness: float = 6.0):
        super().__init__()
        self.feature_dim = o3.Irreps(irreps_in).dim
        self.irreps_in = o3.Irreps(irreps_in)
        self.mix = o3.Linear(self.irreps_in, self.irreps_in)
        self.distance_log_scale = nn.Parameter(torch.tensor(sharpness).log())
        # Lightweight scalar gate: a single linear avoids extra hidden activations.
        self.filter = nn.Linear(1, 1)

    def forward(self, x, edge_index, edge_len):
        if edge_index.numel() == 0:
            return x

        sender, receiver = edge_index
        scale = torch.exp(self.distance_log_scale).to(edge_len.dtype)
        decay = torch.exp(-torch.clamp(edge_len / (scale + 1e-6), min=0.0) ** 2)
        gate = torch.sigmoid(self.filter(edge_len[:, None]).squeeze(-1))
        weights = torch.nan_to_num(decay * gate, nan=0.0, posinf=0.0, neginf=0.0)

        messages = weights[:, None].to(x.dtype) * x[sender]

        if _torch_scatter_available:
            agg = scatter_add(messages, receiver, dim=0, dim_size=x.size(0))
            degree = scatter_add(torch.ones_like(edge_len), receiver, dim=0, dim_size=x.size(0))
        else:
            agg = torch.zeros_like(x)
            agg = agg.index_add(0, receiver, messages)
            degree = torch.zeros(x.size(0), device=x.device, dtype=edge_len.dtype)
            degree = degree.index_add(0, receiver, torch.ones_like(edge_len))

        norm = torch.clamp(degree, min=1.0).unsqueeze(-1)
        agg = agg / norm
        return x + self.mix(agg)

class DenseFlashAttention(nn.Module):
    def __init__(
        self,
        irreps_in,
        hidden_dim,
        num_heads: int = 4,
        message_clip: float | None = None,
        use_conditioned_decay: bool = True,
        share_qkv_mode: str | bool = "none",
        long_range_bins: int = 0,
        debye_init: float = 1.0,
        long_range_heads: int = 1,
        long_range_mix: float = 0.5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = o3.Irreps(irreps_in).dim
        self.message_clip = message_clip
        self.use_conditioned_decay = use_conditioned_decay
        self.long_range_bins = max(0, int(long_range_bins))
        self.long_range_heads = max(0, int(long_range_heads))
        self.long_range_mix = nn.Parameter(torch.tensor(float(long_range_mix)))
        self.debye_kappa = nn.Parameter(torch.tensor(float(debye_init)))
        if isinstance(share_qkv_mode, bool):
            share_qkv_mode = "all" if share_qkv_mode else "none"
        if share_qkv_mode not in {"none", "kv", "all"}:
            raise ValueError("share_qkv_mode must be one of {'none', 'kv', 'all'} or a boolean")
        self.share_qkv_mode = share_qkv_mode

        if self.share_qkv_mode == "all":
            self.w_proj_shared = o3.Linear(irreps_in, irreps_in)
            self.radial_update_shared = o3.Linear(irreps_in, irreps_in)
            self.tangential_update_shared = o3.Linear(irreps_in, irreps_in)
        elif self.share_qkv_mode == "kv":
            self.w_proj = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )
            self.radial_update_shared = o3.Linear(irreps_in, irreps_in)
            self.tangential_update_shared = o3.Linear(irreps_in, irreps_in)
        else:
            self.w_proj = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )
            self.radial_update = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )
            self.tangential_update = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )

        # Geometry-aware scoring vectors
        self.radial_score = nn.Parameter(
            torch.empty(num_heads, self.feature_dim)
        )
        self.tangential_score = nn.Parameter(
            torch.empty(num_heads, self.feature_dim)
        )
        # Use a positive scale so longer bonds are consistently penalized.
        self._radial_distance_log_scale = nn.Parameter(torch.zeros(num_heads))
        # Distance-dependent temperature sharpens radial logits for close
        # neighbors while keeping gradients stable on far bonds.
        self._radial_temp_bias = nn.Parameter(torch.zeros(num_heads))
        self._radial_temp_weight = nn.Parameter(torch.zeros(num_heads))
        # Distance-gated mixer that blends radial/tangential streams so
        # short bonds can emphasize radial updates while long bonds lean on
        # tangential cues without running separate aggregation passes.
        self._mix_bias = nn.Parameter(torch.zeros(num_heads))
        self._mix_scale = nn.Parameter(torch.zeros(num_heads))

        # Environment-conditioned decay/temperature to sharpen locality per site.
        hidden_mid = max(1, self.feature_dim // 2)
        self.radial_decay_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_mid),
                    nn.SiLU(),
                    nn.Linear(hidden_mid, 1),
                )
                for _ in range(num_heads)
            ]
        )
        self.radial_temp_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_mid),
                    nn.SiLU(),
                    nn.Linear(hidden_mid, 1),
                )
                for _ in range(num_heads)
            ]
        )

        self.w_out = o3.Linear(irreps_in, irreps_in)
        if self.long_range_bins > 0:
            lr_heads = max(1, self.long_range_heads)
            self.long_range_bias = nn.Parameter(torch.zeros(lr_heads, self.long_range_bins))
        else:
            self.register_parameter("long_range_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.radial_score)
        nn.init.xavier_uniform_(self.tangential_score)
        nn.init.zeros_(self._radial_distance_log_scale)
        nn.init.zeros_(self._radial_temp_bias)
        nn.init.zeros_(self._radial_temp_weight)
        nn.init.zeros_(self._mix_bias)
        nn.init.zeros_(self._mix_scale)
        def _maybe_reset(layer):
            reset_fn = getattr(layer, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()

        if self.share_qkv_mode == "all":
            for layer in [
                self.w_proj_shared,
                self.radial_update_shared,
                self.tangential_update_shared,
            ]:
                _maybe_reset(layer)
        elif self.share_qkv_mode == "kv":
            for layer in list(self.w_proj) + [
                self.radial_update_shared,
                self.tangential_update_shared,
            ]:
                _maybe_reset(layer)
        else:
            for layer in list(self.w_proj) + list(self.radial_update) + list(self.tangential_update):
                _maybe_reset(layer)
        for mlp in list(self.radial_decay_mlp) + list(self.radial_temp_mlp):
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_vec, edge_len, temperature_scale: float = 1.0, reciprocal_bias=None):
        sender, receiver = edge_index
        num_nodes = x.shape[0]
        if sender.numel() == 0:
            return x

        # These projections are generic node embeddings (queries/keys) rather than
        # energy-specific features. The radial/tangential projections provide the
        # value streams mixed into the updated representation, keeping a clean
        # separation between scoring features and message content.
        if self.share_qkv_mode == "all":
            qk_base = self.w_proj_shared(x)
            radial_base = self.radial_update_shared(x)
            tangential_base = self.tangential_update_shared(x)
            qk_proj = qk_base.unsqueeze(0).expand(self.num_heads, -1, -1)
            radial_proj = radial_base.unsqueeze(0).expand(self.num_heads, -1, -1)
            tangential_proj = tangential_base.unsqueeze(0).expand(self.num_heads, -1, -1)
        elif self.share_qkv_mode == "kv":
            qk_proj = torch.stack([layer(x) for layer in self.w_proj], dim=0)
            radial_base = self.radial_update_shared(x)
            tangential_base = self.tangential_update_shared(x)
            radial_proj = radial_base.unsqueeze(0).expand(self.num_heads, -1, -1)
            tangential_proj = tangential_base.unsqueeze(0).expand(self.num_heads, -1, -1)
        else:
            qk_proj = torch.stack([layer(x) for layer in self.w_proj], dim=0)
            radial_proj = torch.stack([layer(x) for layer in self.radial_update], dim=0)
            tangential_proj = torch.stack([layer(x) for layer in self.tangential_update], dim=0)

        out = self._geometric_decomposition_attention(
            qk_proj,
            radial_proj,
            tangential_proj,
            sender,
            receiver,
            edge_len,
            temperature_scale=temperature_scale,
            reciprocal_bias=reciprocal_bias,
        )

        out = torch.nan_to_num(out)
        return x + self.w_out(out)

    def _geometric_decomposition_attention(
        self,
        qk_proj,
        radial_proj,
        tangential_proj,
        sender,
        receiver,
        edge_len,
        temperature_scale: float,
        reciprocal_bias=None,
    ):
        # *_proj are shaped (num_heads, num_nodes, feature_dim)
        num_heads = qk_proj.shape[0]
        num_nodes = qk_proj.shape[1]

        # qk_proj drives the scoring (queries/keys) while radial/tangential
        # projections carry the values that get mixed into the updated features.
        qk_delta = qk_proj[:, sender] - qk_proj[:, receiver]
        radial_delta = radial_proj[:, sender] - radial_proj[:, receiver]
        tangential_delta = tangential_proj[:, sender] - tangential_proj[:, receiver]

        # Radial energy penalizes long bonds, tangential is distance agnostic.
        radial_distance_scale = F.softplus(self._radial_distance_log_scale).to(edge_len.dtype)[:, None]

        receiver_feat = qk_proj[:, receiver]  # (heads, edges, feature_dim)
        if self.use_conditioned_decay:
            decay_offset = torch.stack(
                [mlp(receiver_feat[h]).squeeze(-1) for h, mlp in enumerate(self.radial_decay_mlp)],
                dim=0,
            )
            temp_offset = torch.stack(
                [mlp(receiver_feat[h]).squeeze(-1) for h, mlp in enumerate(self.radial_temp_mlp)],
                dim=0,
            )
        else:
            decay_offset = torch.zeros_like(qk_delta[..., 0])
            temp_offset = torch.zeros_like(qk_delta[..., 0])

        radial_logits = (
            (qk_delta * self.radial_score[:, None, :]).sum(dim=-1).float()
            - (radial_distance_scale + decay_offset).float() * edge_len.float()
            - F.softplus(self.debye_kappa).float() * edge_len.float()
        )
        if reciprocal_bias is not None and self.long_range_bias is not None:
            # reciprocal_bias: (E, B)
            rb = reciprocal_bias
            lr_term = torch.einsum("hb,eb->he", self.long_range_bias, rb)
            mix = torch.clamp(torch.sigmoid(self.long_range_mix), min=0.0, max=1.0)
            radial_logits = radial_logits + mix * lr_term
        radial_temp = F.softplus(
            self._radial_temp_bias[:, None]
            + self._radial_temp_weight[:, None] * edge_len
            + temp_offset
        )
        radial_temp = (radial_temp * temperature_scale).float()
        radial_logits = radial_logits / (radial_temp + 1e-4)
        tangential_logits = (
            qk_delta * self.tangential_score[:, None, :]
        ).sum(dim=-1).float()
        if reciprocal_bias is not None and self.long_range_bias is not None:
            lr_term = torch.einsum("hb,eb->he", self.long_range_bias, reciprocal_bias)
            mix = torch.clamp(torch.sigmoid(self.long_range_mix), min=0.0, max=1.0)
            tangential_logits = tangential_logits + mix * lr_term

        num_edges = sender.numel()
        feature_dim = qk_proj.shape[-1]
        out = torch.zeros_like(qk_proj)
        if num_edges == 0:
            return out.mean(dim=0)

        expanded_receiver = receiver.unsqueeze(0).expand(num_heads, -1)

        def _segment_softmax(logits):
            # logits: (heads, num_edges)
            if _torch_scatter_available:
                return scatter_softmax(
                    logits, expanded_receiver, dim=1, dim_size=num_nodes
                )

            max_init = torch.full(
                (num_heads, num_nodes),
                float("-inf"),
                device=logits.device,
                dtype=logits.dtype,
            )
            max_per_node = max_init.scatter_reduce(
                1,
                expanded_receiver,
                logits,
                reduce="amax",
                include_self=True,
            )
            max_per_node = torch.where(
                torch.isfinite(max_per_node), max_per_node, torch.zeros_like(max_per_node)
            )

            centered = logits - max_per_node.gather(1, expanded_receiver)
            exp_logits = torch.exp(centered)

            denom = torch.zeros_like(max_per_node).scatter_add(
                1, expanded_receiver, exp_logits
            )
            alpha = exp_logits / (denom.gather(1, expanded_receiver) + 1e-9)
            return torch.nan_to_num(alpha)

        radial_alpha = torch.nan_to_num(_segment_softmax(radial_logits))
        tangential_alpha = torch.nan_to_num(_segment_softmax(tangential_logits))

        mix_gate = torch.sigmoid(
            self._mix_bias[:, None]
            + self._mix_scale[:, None] * edge_len[None, :]
        )

        blended_alpha = mix_gate * radial_alpha + (1.0 - mix_gate) * tangential_alpha
        blended_delta = mix_gate[:, :, None] * radial_delta + (
            1.0 - mix_gate[:, :, None]
        ) * tangential_delta

        weighted_delta = blended_alpha[..., None].to(blended_delta.dtype) * blended_delta
        if self.message_clip is not None:
            clip = torch.tensor(self.message_clip, device=weighted_delta.device, dtype=weighted_delta.dtype)
            norms = weighted_delta.norm(dim=-1, keepdim=True)
            safe = norms + 1e-8
            scale = torch.tanh(norms / clip) * (clip / safe)
            weighted_delta = weighted_delta * scale
        if _torch_scatter_available:
            out = scatter_add(
                weighted_delta,
                expanded_receiver[:, :, None].expand(-1, -1, feature_dim),
                dim=1,
                dim_size=num_nodes,
            )
        else:
            out = out.scatter_add(
                1,
                expanded_receiver[:, :, None].expand(-1, -1, feature_dim),
                weighted_delta,
            )
        return out.mean(dim=0)
