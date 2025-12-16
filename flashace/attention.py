import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

class DenseFlashAttention(nn.Module):
    def __init__(self, irreps_in, hidden_dim):
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.w_q = o3.Linear(irreps_in, irreps_in)
        self.w_k = o3.Linear(irreps_in, irreps_in)
        self.w_v = o3.Linear(irreps_in, irreps_in)
        self.w_out = o3.Linear(irreps_in, irreps_in)

    def forward(self, x, edge_index):
        sender, receiver = edge_index
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        num_nodes = x.shape[0]
        if sender.numel() == 0:
            return x

        # Preallocate neighbor tensors (padded) so we can run fused
        # scaled_dot_product_attention calls that leverage Flash Attention
        # kernels when available on the current device.
        deg = torch.bincount(receiver, minlength=num_nodes)
        max_deg = int(deg.max().item())
        head_dim = Q.shape[-1]

        k_buf = torch.zeros((num_nodes, max_deg, head_dim), device=x.device, dtype=K.dtype)
        v_buf = torch.zeros_like(k_buf)

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

        k_buf[receiver, pos] = K[sender]
        v_buf[receiver, pos] = V[sender]

        out = torch.zeros_like(Q)
        unique_deg = torch.unique(deg)

        # Avoid passing an attention mask so Flash kernels can be selected; we
        # instead slice each degree bucket to exclude padding entirely.
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            for d in unique_deg.tolist():
                if d == 0:
                    continue

                idx = deg == d
                q = Q[idx][:, None, None, :]
                k = k_buf[idx][:, None, :d, :]
                v = v_buf[idx][:, None, :d, :]

                out_bucket = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    is_causal=False,
                    scale=self.scale,
                )

                out[idx] = out_bucket.squeeze(2).squeeze(1).to(out.dtype)

        out = torch.nan_to_num(out)
        return x + self.w_out(out)
