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

        # Preallocate neighbor tensors (padded) so we can run a single fused
        # scaled_dot_product_attention call that leverages Flash Attention
        # kernels when available on the current device.
        deg = torch.bincount(receiver, minlength=num_nodes)
        max_deg = int(deg.max().item())
        head_dim = Q.shape[-1]

        k_buf = torch.zeros((num_nodes, max_deg, head_dim), device=x.device, dtype=K.dtype)
        v_buf = torch.zeros_like(k_buf)
        valid = torch.zeros((num_nodes, max_deg), device=x.device, dtype=torch.bool)

        # Track the next free slot for each receiver node.
        fill_ptr = torch.zeros(num_nodes, device=x.device, dtype=torch.long)
        for s, r in zip(sender, receiver):
            idx = fill_ptr[r]
            k_buf[r, idx] = K[s]
            v_buf[r, idx] = V[s]
            valid[r, idx] = True
            fill_ptr[r] = idx + 1

        q = Q[:, None, None, :]
        k = k_buf[:, None, :, :]
        v = v_buf[:, None, :, :]
        attn_mask = ~valid[:, None, None, :]

        # Prefer Flash Attention kernels while avoiding the mem-efficient
        # backward path (which lacks derivatives on some builds) and allowing
        # math fallback when flash is unavailable (e.g., CPU execution).
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=True,
        ):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                scale=self.scale,
            )

        out = torch.nan_to_num(out.squeeze(2).squeeze(1))
        return x + self.w_out(out)
