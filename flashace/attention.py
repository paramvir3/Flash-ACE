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

        self.w_out = o3.Linear(irreps_in, irreps_in)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.radial_score)
        nn.init.xavier_uniform_(self.tangential_score)
        nn.init.zeros_(self._radial_distance_log_scale)
        nn.init.zeros_(self._radial_temp_bias)
        nn.init.zeros_(self._radial_temp_weight)

    def forward(self, x, edge_index, edge_vec, edge_len):
        sender, receiver = edge_index
        num_nodes = x.shape[0]
        if sender.numel() == 0:
            return x

        deg = torch.bincount(receiver, minlength=num_nodes)
        # Vectorized sorting of edges by receiver so we can build per-degree
        # buckets without a Python loop.
        order = torch.argsort(receiver)
        # Start index for each receiver in the sorted list of edges.
        start_per_receiver = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=x.device, dtype=deg.dtype),
                deg.to(torch.long)
            ]),
            dim=0,
        )[:-1]

        energy_proj = torch.stack([layer(x) for layer in self.w_proj], dim=0)
        radial_proj = torch.stack([layer(x) for layer in self.radial_update], dim=0)
        tangential_proj = torch.stack([layer(x) for layer in self.tangential_update], dim=0)

        out = self._geometric_decomposition_attention(
            energy_proj,
            radial_proj,
            tangential_proj,
            deg,
            sender,
            receiver,
            order,
            start_per_receiver,
            edge_len,
        )

        out = torch.nan_to_num(out)
        return x + self.w_out(out)

    def _geometric_decomposition_attention(
        self,
        energy_proj,
        radial_proj,
        tangential_proj,
        deg,
        sender,
        receiver,
        order,
        start_per_receiver,
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

        out = torch.zeros_like(energy_proj)
        unique_deg = torch.unique(deg)

        for d in unique_deg.tolist():
            if d == 0:
                continue

            idx = deg == d
            if not torch.any(idx):
                continue

            # Gather contiguous edge blocks for this degree directly from the
            # sorted edge list so we avoid constructing large zero-padded
            # buffers when max_deg is much larger than the typical degree.
            edge_offsets = start_per_receiver[idx]
            gather_range = (
                edge_offsets[:, None]
                + torch.arange(d, device=edge_offsets.device)
            )
            edge_ids = order[gather_range]

            radial_slice = radial_logits[:, edge_ids]
            tangential_slice = tangential_logits[:, edge_ids]
            radial_delta_slice = radial_delta[:, edge_ids]
            tangential_delta_slice = tangential_delta[:, edge_ids]

            radial_alpha = torch.softmax(radial_slice, dim=2)
            tangential_alpha = torch.softmax(tangential_slice, dim=2)

            radial_msg = torch.einsum(
                "hnd,hndf->hnf", radial_alpha, radial_delta_slice
            )
            tangential_msg = torch.einsum(
                "hnd,hndf->hnf", tangential_alpha, tangential_delta_slice
            )

            out[:, idx, :] = (radial_msg + tangential_msg).to(out.dtype)

        return out.mean(dim=0)
