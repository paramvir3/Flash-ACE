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
        max_deg = int(deg.max().item())
        if max_deg == 0:
            return out.mean(dim=0)

        # Build a dense gather map of shape (num_nodes, max_deg) so the per-head
        # attention computation can run as a single fused softmax/einsum instead
        # of looping over degrees. Positions beyond a node's actual degree are
        # masked out with -inf logits and zeroed messages.
        gather_range = torch.arange(max_deg, device=order.device)
        edge_ids = start_per_receiver[:, None] + gather_range
        valid = gather_range[None, :] < deg[:, None]

        # Clamp invalid positions to a safe index and apply masks before
        # softmax so no probability mass leaks into padded slots.
        safe_edge_ids = torch.where(
            valid, edge_ids, torch.zeros_like(edge_ids)
        )
        edge_ids = order[safe_edge_ids]

        # (heads, num_nodes, max_deg)
        valid_exp = valid.unsqueeze(0)

        # Direct indexing avoids expanding logits to (heads, num_nodes, num_edges)
        # which was a major memory/time bottleneck. Shape: (heads, num_nodes, max_deg).
        radial_slice = radial_logits[:, edge_ids]
        tangential_slice = tangential_logits[:, edge_ids]

        radial_slice = torch.where(
            valid_exp, radial_slice, torch.full_like(radial_slice, float("-inf"))
        )
        tangential_slice = torch.where(
            valid_exp,
            tangential_slice,
            torch.full_like(tangential_slice, float("-inf")),
        )

        radial_alpha = torch.softmax(radial_slice, dim=2)
        tangential_alpha = torch.softmax(tangential_slice, dim=2)
        radial_alpha = torch.where(
            valid_exp, radial_alpha, torch.zeros_like(radial_alpha)
        )
        tangential_alpha = torch.where(
            valid_exp, tangential_alpha, torch.zeros_like(tangential_alpha)
        )
        radial_alpha = torch.nan_to_num(radial_alpha)
        tangential_alpha = torch.nan_to_num(tangential_alpha)

        radial_delta_slice = radial_delta[:, edge_ids]
        tangential_delta_slice = tangential_delta[:, edge_ids]

        radial_delta_slice = torch.where(
            valid_exp[..., None], radial_delta_slice, torch.zeros_like(radial_delta_slice)
        )
        tangential_delta_slice = torch.where(
            valid_exp[..., None],
            tangential_delta_slice,
            torch.zeros_like(tangential_delta_slice),
        )

        radial_msg = torch.einsum("hnd,hndf->hnf", radial_alpha, radial_delta_slice)
        tangential_msg = torch.einsum(
            "hnd,hndf->hnf", tangential_alpha, tangential_delta_slice
        )

        out = radial_msg + tangential_msg
        return out.mean(dim=0)
