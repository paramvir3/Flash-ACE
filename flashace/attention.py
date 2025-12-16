import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

class DenseFlashAttention(nn.Module):
    def __init__(self, irreps_in, hidden_dim, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = o3.Irreps(irreps_in).dim

        # Content projections for energy scores.
        self.w_proj = nn.ModuleList(
            [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
        )
        # Separate update projections so radial/tangential channels can learn
        # distinct filters rather than reusing the energy encoder.
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
        self._radial_distance_log_scale = nn.Parameter(torch.tensor(0.0))
        # Distance-dependent temperature sharpens radial logits for close
        # neighbors while keeping gradients stable on far bonds.
        self._radial_temp_bias = nn.Parameter(torch.zeros(num_heads))
        self._radial_temp_weight = nn.Parameter(torch.zeros(num_heads))
        # Distance-gated mixer that blends radial/tangential streams so
        # short bonds can emphasize radial updates while long bonds lean on
        # tangential cues without running separate aggregation passes.
        self._mix_bias = nn.Parameter(torch.zeros(num_heads))
        self._mix_scale = nn.Parameter(torch.zeros(num_heads))

        self.w_out = o3.Linear(irreps_in, irreps_in)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.radial_score)
        nn.init.xavier_uniform_(self.tangential_score)
        nn.init.zeros_(self._radial_distance_log_scale)
        nn.init.zeros_(self._radial_temp_bias)
        nn.init.zeros_(self._radial_temp_weight)
        nn.init.zeros_(self._mix_bias)
        nn.init.zeros_(self._mix_scale)

    def forward(self, x, edge_index, edge_vec, edge_len):
        sender, receiver = edge_index
        num_nodes = x.shape[0]
        if sender.numel() == 0:
            return x

        energy_proj = torch.stack([layer(x) for layer in self.w_proj], dim=0)
        radial_proj = torch.stack([layer(x) for layer in self.radial_update], dim=0)
        tangential_proj = torch.stack([layer(x) for layer in self.tangential_update], dim=0)

        out = self._geometric_decomposition_attention(
            energy_proj,
            radial_proj,
            tangential_proj,
            sender,
            receiver,
            edge_len,
        )

        out = torch.nan_to_num(out)
        return x + self.w_out(out)

    def _geometric_decomposition_attention(
        self,
        energy_proj,
        radial_proj,
        tangential_proj,
        sender,
        receiver,
        edge_len,
    ):
        # *_proj are shaped (num_heads, num_nodes, feature_dim)
        num_heads = energy_proj.shape[0]
        num_nodes = energy_proj.shape[1]

        energy_delta = energy_proj[:, sender] - energy_proj[:, receiver]
        radial_delta = radial_proj[:, sender] - radial_proj[:, receiver]
        tangential_delta = tangential_proj[:, sender] - tangential_proj[:, receiver]

        # Radial energy penalizes long bonds, tangential is distance agnostic.
        radial_distance_scale = F.softplus(self._radial_distance_log_scale)
        radial_logits = (
            (energy_delta * self.radial_score[:, None, :]).sum(dim=-1)
            - radial_distance_scale * edge_len
        )
        radial_temp = F.softplus(
            self._radial_temp_bias[:, None]
            + self._radial_temp_weight[:, None] * edge_len
        )
        radial_logits = radial_logits / (radial_temp + 1e-4)
        tangential_logits = (
            energy_delta * self.tangential_score[:, None, :]
        ).sum(dim=-1)

        num_edges = sender.numel()
        feature_dim = energy_proj.shape[-1]
        out = torch.zeros_like(energy_proj)
        if num_edges == 0:
            return out.mean(dim=0)

        expanded_receiver = receiver.unsqueeze(0).expand(num_heads, -1)

        def _segment_softmax(logits):
            # logits: (heads, num_edges)
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

        radial_alpha = _segment_softmax(radial_logits)
        tangential_alpha = _segment_softmax(tangential_logits)

        mix_gate = torch.sigmoid(
            self._mix_bias[:, None]
            + self._mix_scale[:, None] * edge_len[None, :]
        )

        blended_alpha = mix_gate * radial_alpha + (1.0 - mix_gate) * tangential_alpha
        blended_delta = mix_gate[:, :, None] * radial_delta + (
            1.0 - mix_gate[:, :, None]
        ) * tangential_delta

        weighted_delta = blended_alpha[..., None] * blended_delta
        out = out.scatter_add(
            1,
            expanded_receiver[:, :, None].expand(-1, -1, feature_dim),
            weighted_delta,
        )
        return out.mean(dim=0)
