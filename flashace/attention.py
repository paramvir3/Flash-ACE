import torch
import torch.nn as nn
from e3nn import o3

class DenseFlashAttention(nn.Module):
    def __init__(self, irreps_in, hidden_dim, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = o3.Irreps(irreps_in).dim

        # Content projections
        self.w_proj = nn.ModuleList(
            [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
        )

        # Geometry-aware scoring vectors
        self.radial_score = nn.Parameter(
            torch.empty(num_heads, self.feature_dim)
        )
        self.tangential_score = nn.Parameter(
            torch.empty(num_heads, self.feature_dim)
        )
        self.radial_distance_scale = nn.Parameter(torch.tensor(1.0))

        self.w_out = o3.Linear(irreps_in, irreps_in)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.radial_score)
        nn.init.xavier_uniform_(self.tangential_score)
        nn.init.constant_(self.radial_distance_scale, 1.0)

    def forward(self, x, edge_index, edge_vec, edge_len):
        sender, receiver = edge_index
        num_nodes = x.shape[0]
        if sender.numel() == 0:
            return x

        deg = torch.bincount(receiver, minlength=num_nodes)
        max_deg = int(deg.max().item())

        # Vectorized position within each receiver bucket so we avoid a Python
        # loop over edges, which becomes a bottleneck on large graphs.
        order = torch.argsort(receiver)
        receiver_sorted = receiver[order]

        # Start index for each receiver in the sorted list of edges.
        start_per_receiver = torch.cumsum(
            torch.cat([
                torch.zeros(1, device=x.device, dtype=deg.dtype),
                deg.to(torch.long)
            ]),
            dim=0,
        )[:-1]

        pos_sorted = torch.arange(sender.numel(), device=x.device) - start_per_receiver[receiver_sorted]

        pos = torch.empty_like(order)
        pos[order] = pos_sorted

        stacked_proj = torch.stack([layer(x) for layer in self.w_proj], dim=0)

        out = self._geometric_decomposition_attention(
            stacked_proj,
            deg,
            pos,
            max_deg,
            sender,
            receiver,
            edge_len,
        )

        out = torch.nan_to_num(out)
        return x + self.w_out(out)

    def _geometric_decomposition_attention(
        self,
        proj,
        deg,
        pos,
        max_deg,
        sender,
        receiver,
        edge_len,
    ):
        # proj is shaped (num_heads, num_nodes, feature_dim)
        num_heads = proj.shape[0]
        num_nodes = proj.shape[1]

        delta = proj[:, sender] - proj[:, receiver]

        delta_buf = torch.zeros(
            (num_heads, num_nodes, max_deg, self.feature_dim),
            device=proj.device,
            dtype=proj.dtype,
        )
        radial_energy = torch.full(
            (num_heads, num_nodes, max_deg),
            fill_value=-torch.finfo(proj.dtype).max,
            device=proj.device,
            dtype=proj.dtype,
        )
        tangential_energy = torch.full_like(radial_energy, -torch.finfo(proj.dtype).max)

        delta_buf[:, receiver, pos] = delta

        # Radial energy penalizes long bonds, tangential is distance agnostic.
        radial_logits = (
            (delta * self.radial_score[:, None, :]).sum(dim=-1)
            - self.radial_distance_scale * edge_len
        )
        tangential_logits = (delta * self.tangential_score[:, None, :]).sum(dim=-1)

        radial_energy[:, receiver, pos] = radial_logits
        tangential_energy[:, receiver, pos] = tangential_logits

        out = torch.zeros_like(proj)
        unique_deg = torch.unique(deg)

        for d in unique_deg.tolist():
            if d == 0:
                continue

            idx = deg == d
            delta_slice = delta_buf[:, idx, :d, :]
            radial_slice = radial_energy[:, idx, :d]
            tangential_slice = tangential_energy[:, idx, :d]

            radial_alpha = torch.softmax(radial_slice, dim=2)
            tangential_alpha = torch.softmax(tangential_slice, dim=2)

            radial_msg = torch.einsum("hnd,hndf->hnf", radial_alpha, delta_slice)
            tangential_msg = torch.einsum("hnd,hndf->hnf", tangential_alpha, delta_slice)

            out[:, idx, :] = (radial_msg + tangential_msg).to(out.dtype)

        return out.mean(dim=0)
