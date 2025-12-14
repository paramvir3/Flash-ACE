import torch
from flashace.physics import ACE_Descriptor


def test_ace_descriptor_uses_node_attributes():
    torch.manual_seed(0)
    ace = ACE_Descriptor(r_max=5.0, l_max=2, num_radial=4, hidden_dim=8)

    node_attrs = torch.randn(3, 8)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.5, 0.5, 0.0]]
    )

    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_len = torch.norm(edge_vec, dim=1)

    baseline = ace(node_attrs, edge_index, edge_vec, edge_len)

    modified_attrs = node_attrs.clone()
    modified_attrs[0] += 1.0  # change chemical identity-like channel
    altered = ace(modified_attrs, edge_index, edge_vec, edge_len)

    assert not torch.allclose(baseline, altered)
    assert baseline.shape == altered.shape == (node_attrs.shape[0], ace.irreps_out.dim)
