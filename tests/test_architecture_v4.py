import numpy as np
import torch

from flashace.model import TransformersACEV4
from transformers_ace.aot import make_aot_compatible
from transformers_ace.deploy import LAMMPSEnergyModel


def _model():
    torch.manual_seed(31)
    return TransformersACEV4(
        r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
        correlation_order=4, correlation_channels=4, attention_num_heads=2,
        attention_num_shells=3,
    ).eval()


def _data(positions):
    n_atoms = positions.shape[0]
    senders, receivers = [], []
    for receiver in range(n_atoms):
        for sender in range(n_atoms):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    return {
        "z": torch.tensor([8, 1, 1, 8][:n_atoms]),
        "pos": positions,
        "cell": torch.eye(3) * 9.0,
        "edge_index": torch.tensor([senders, receivers]),
        "edge_shift": torch.zeros((len(senders), 3)),
        "volume": torch.tensor(729.0),
    }


def test_v4_distinct_pair_token_removes_self_contractions_exactly():
    model = _model()
    data = _data(torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]))
    edge_vec = data["pos"][data["edge_index"][0]] - data["pos"][data["edge_index"][1]]
    edge_len = torch.linalg.norm(edge_vec, dim=1)
    density, edges, _ = model.ace._density(model.emb(data["z"]), data["edge_index"], edge_vec, edge_len)
    pair = model.ace.contractions[0](density, density)
    edge_self = model.ace.contractions[0](edges, edges)
    self_sum = torch.zeros_like(pair)
    self_sum.index_add_(0, data["edge_index"][1], edge_self)
    assert torch.isfinite(pair - self_sum).all()
    assert not torch.allclose(pair, pair - self_sum)


def test_v4_is_equivariant_and_has_no_sender_state_dependency():
    model = _model()
    positions = torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]])
    energy, forces, _, _ = model(_data(positions), training=False, compute_stress=False)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_energy, rotated_forces, _, _ = model(_data(positions @ rotation.T), training=False, compute_stress=False)
    np.testing.assert_allclose(float(rotated_energy.detach()), float(energy.detach()), rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(rotated_forces.detach(), (forces @ rotation.T).detach(), rtol=5e-4, atol=5e-4)

    layer = model.layers[0]
    x = torch.randn(3, model.attention_irreps.dim)
    tokens = torch.randn(2, model.ace.irreps_correlation.dim)
    receiver = torch.tensor([0, 0])
    length = torch.tensor([1.0, 1.3])
    cutoff = model.ace.cutoff(length)
    kind = torch.zeros(2, dtype=torch.long)
    baseline = layer(x, tokens, receiver, length, cutoff, kind)
    x[1:] += 100.0
    torch.testing.assert_close(baseline[0], layer(x, tokens, receiver, length, cutoff, kind)[0])


def test_v4_symmetric_stress_and_aot_primitives_agree():
    model = _model()
    data = _data(torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]))
    _, _, stress, _ = model(data, training=False, compute_stress=True)
    assert torch.isfinite(stress).all()
    torch.testing.assert_close(stress, stress.T)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference = LAMMPSEnergyModel(model)
    converted = LAMMPSEnergyModel(make_aot_compatible(model))
    args = (data["z"], data["pos"], data["cell"], data["edge_index"], data["edge_shift"], torch.zeros(6), torch.ones(4))
    torch.testing.assert_close(converted(*args), reference(*args), atol=5e-6, rtol=5e-6)


def test_v4_correlation_rank_masks_whole_irrep_copies():
    model = _model()
    model.set_correlation_rank(1)
    mask = model.ace.correlation_channel_mask
    for (multiplicity, irrep), part in zip(model.ace.irreps_correlation, model.ace.irreps_correlation.slices()):
        block = mask[part].reshape(multiplicity, irrep.dim)
        assert torch.all(block == block[:, :1])
        assert block[:, 0].sum() >= 1
