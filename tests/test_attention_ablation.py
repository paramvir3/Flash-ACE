"""The ablation arms must be valid models, and must actually differ.

An ablation is only meaningful if every arm is a physically correct potential
(otherwise a control is handicapped by a broken symmetry rather than by the
missing mechanism) and if the arms are genuinely different functions (otherwise
the comparison is vacuous).
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from flashace.model import TransformersACE

R_MAX = 6.0
ARMS = ["full", "no_qk", "uniform", "none"]
POSITIONS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [2.1, 0.2, 0.1],
        [0.3, 2.4, 0.4],
        [1.1, 1.2, 2.6],
        [-2.2, 0.9, 1.1],
    ],
    dtype=torch.float64,
)
NUMBERS = torch.tensor([55, 82, 53, 53, 53])


def _edges(positions):
    senders, receivers = [], []
    for receiver in range(len(positions)):
        for sender in range(len(positions)):
            if sender != receiver and torch.linalg.norm(
                positions[sender] - positions[receiver]
            ) < R_MAX:
                senders.append(sender)
                receivers.append(receiver)
    return torch.tensor([senders, receivers], dtype=torch.long)


def _model(mode, randomise=True, ffn_hidden=None):
    torch.manual_seed(0)
    model = TransformersACE(
        r_max=R_MAX,
        l_max=2,
        num_radial=12,
        hidden_dim=64,
        num_layers=1,
        correlation_order=4,
        correlation_channels=16,
        attention_num_heads=2,
        attention_mode=mode,
        attention_ffn_hidden=ffn_hidden,
    ).double()
    if randomise:
        # Near initialisation the arms can agree by accident; push the weights
        # away from it so the tests are not vacuous.
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.dim() >= 1:
                    parameter.copy_(torch.randn_like(parameter) * 0.5)
    return model.eval()


def _run(model, positions):
    data = {
        "z": NUMBERS,
        "pos": positions.clone().requires_grad_(True),
        "edge_index": _edges(positions),
        "volume": torch.tensor(1000.0, dtype=torch.float64),
    }
    return model(data, training=False, compute_stress=False)


@pytest.mark.parametrize("mode", ARMS)
def test_every_arm_is_rotation_equivariant(mode):
    model = _model(mode)
    energy, forces, _, _ = _run(model, POSITIONS)
    for _ in range(3):
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated_energy, rotated_forces, _, _ = _run(model, POSITIONS @ rotation.T)
        np.testing.assert_allclose(float(rotated_energy), float(energy), rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            rotated_forces.detach().numpy(),
            (forces @ rotation.T).detach().numpy(),
            rtol=0,
            atol=1e-9,
        )


@pytest.mark.parametrize("mode", ARMS)
def test_every_arm_has_conservative_forces(mode):
    model = _model(mode)
    _, forces, _, _ = _run(model, POSITIONS)
    step = 1e-3
    for atom, axis in ((1, 0), (3, 2)):
        forward = POSITIONS.clone(); forward[atom, axis] += step
        backward = POSITIONS.clone(); backward[atom, axis] -= step
        finite = -(float(_run(model, forward)[0]) - float(_run(model, backward)[0])) / (2 * step)
        assert abs(finite - float(forces[atom, axis])) < 1e-4


@pytest.mark.parametrize("mode", ARMS)
def test_unused_projections_are_not_allocated(mode):
    """Each arm must report the parameter count it actually uses."""
    block = _model(mode, randomise=False).layers[0]
    if mode == "full":
        assert block.q_proj is not None and block.k_proj is not None
    else:
        assert block.q_proj is None and block.k_proj is None
    if mode in ("uniform", "none"):
        assert block.radial_bias is None
    if mode == "none":
        assert block.value_proj is None and block.out_proj is None


def test_arms_are_ordered_by_parameter_count_as_documented():
    counts = {
        mode: sum(p.numel() for p in _model(mode, randomise=False).parameters())
        for mode in ARMS
    }
    assert counts["full"] > counts["no_qk"] > counts["uniform"] > counts["none"]
    # The query-key projections are a small share, which is what makes the
    # comparison nearly capacity-matched by construction.
    assert (counts["full"] - counts["no_qk"]) / counts["full"] < 0.05


def test_parameter_matched_control_matches_full_capacity():
    """The referee-proof arm: same parameter count, no query-key term."""
    full = sum(p.numel() for p in _model("full", randomise=False).parameters())
    matched = sum(
        p.numel()
        for p in _model("no_qk", randomise=False, ffn_hidden=157).parameters()
    )
    assert abs(matched - full) / full < 0.001


def test_only_full_attention_can_distinguish_equidistant_neighbours():
    """The mechanism check: same distance, different species.

    A distance-only filter cannot separate two neighbours at equal range. If
    `full` could not either, the ablation would be comparing identical
    functions and no result from it would mean anything.
    """
    import flashace.model as model_module

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [-2.5, 0.0, 0.0], [0.0, 3.1, 0.0]],
        dtype=torch.float64,
    )
    numbers = torch.tensor([55, 82, 53, 53])

    def weights(mode):
        model = _model(mode)
        captured = {}
        original = model_module._segment_cutoff_softmax

        def spy(logits, index, cutoff, num_nodes):
            alpha = original(logits, index, cutoff, num_nodes)
            captured["alpha"] = alpha.detach().clone()
            captured["index"] = index.clone()
            return alpha

        model_module._segment_cutoff_softmax = spy
        try:
            edge_index = _edges(positions)
            model(
                {
                    "z": numbers,
                    "pos": positions.clone().requires_grad_(True),
                    "edge_index": edge_index,
                    "volume": torch.tensor(1000.0, dtype=torch.float64),
                },
                training=False,
                compute_stress=False,
            )
        finally:
            model_module._segment_cutoff_softmax = original
        alpha, index = captured["alpha"], captured["index"]
        into_centre = index == 0
        senders = edge_index[0][into_centre]
        head0 = alpha[into_centre][:, 0]
        return {int(s): float(w) for s, w in zip(senders, head0) if int(s) in (1, 2)}

    full = weights("full")
    assert abs(full[1] - full[2]) > 1e-6, "full attention cannot resolve species"
    for mode in ("no_qk", "uniform"):
        control = weights(mode)
        assert control[1] == control[2], f"{mode} unexpectedly depends on content"
