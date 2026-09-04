"""Regression tests for the P3/D2/D3/D4 defect fixes.

Each test pins one property that was previously violated. They are written so
that reintroducing the original defect makes them fail, not merely so that the
current code passes.
"""

import inspect

import numpy as np
import pytest
import torch
from e3nn import o3

import train as train_module
from flashace.model import (
    TransformersACE,
    _irrepwise_layer_scale,
    _layer_scale_num_copies,
)
from train import MetricTracker


R_MAX = 4.0
POSITIONS = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.9, 0.2, 0.1],
        [0.3, 2.1, 0.4],
        [1.1, 1.2, 2.3],
        [-1.6, 0.9, 1.1],
    ],
    dtype=torch.float64,
)
NUMBERS = [55, 82, 53, 53, 53]


def _edges(positions, cutoff=R_MAX):
    senders, receivers = [], []
    for receiver in range(len(positions)):
        for sender in range(len(positions)):
            if sender == receiver:
                continue
            if torch.linalg.norm(positions[sender] - positions[receiver]) < cutoff:
                senders.append(sender)
                receivers.append(receiver)
    return torch.tensor([senders, receivers], dtype=torch.long)


def _model(dropout=0.0, seed=0):
    torch.manual_seed(seed)
    model = TransformersACE(
        r_max=R_MAX,
        l_max=2,
        num_radial=6,
        hidden_dim=16,
        num_layers=1,
        correlation_order=4,
        correlation_channels=8,
        attention_num_heads=2,
        attention_dropout=dropout,
    ).double()
    model.eval()
    return model


def _run(model, positions):
    data = {
        "z": torch.tensor(NUMBERS),
        "pos": positions.clone(),
        "edge_index": _edges(positions),
        "volume": torch.tensor(1000.0, dtype=torch.float64),
    }
    return model(data, training=False, compute_stress=False)


# --------------------------------------------------------------------------
# P3: the non-equivariant layer-scale state must be unrepresentable
# --------------------------------------------------------------------------

def test_layer_scale_is_stored_per_irrep_copy_not_per_component():
    """The parameter must have one entry per irrep copy.

    Storing one entry per *component* is what allowed a trained checkpoint to
    hold a rotationally inconsistent scale that the forward pass then silently
    projected away.
    """
    block = _model().layers[0]
    irreps = o3.Irreps(block.node_irreps)
    n_copies = sum(multiplicity for multiplicity, _ in irreps)

    assert block.layer_scale_attn.numel() == n_copies
    assert block.layer_scale_attn.numel() < irreps.dim


def test_layer_scale_broadcast_is_constant_within_every_irrep_copy():
    block = _model().layers[0]
    with torch.no_grad():
        block.layer_scale_attn.copy_(
            torch.linspace(-0.9, 1.3, block.layer_scale_attn.numel(), dtype=torch.float64)
        )

    expanded = _irrepwise_layer_scale(block.layer_scale_attn, block._layer_scale_blocks)
    assert expanded.numel() == o3.Irreps(block.node_irreps).dim

    offset = 0
    for multiplicity, irrep_dim in block._layer_scale_blocks:
        width = multiplicity * irrep_dim
        copies = expanded[offset : offset + width].reshape(multiplicity, irrep_dim)
        spread = (copies.max(dim=-1).values - copies.min(dim=-1).values).max()
        assert float(spread) == 0.0
        offset += width


def test_energy_is_rotation_invariant_for_adversarial_layer_scales():
    """Equivariance must hold for *any* parameter values, not just near init."""
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() >= 1:
                parameter.copy_(torch.randn_like(parameter) * 0.7)

    energy, forces, _, _ = _run(model, POSITIONS)
    for _ in range(4):
        # Build the rotation in float64. A float32 matrix cast to float64 is only
        # orthogonal to ~1e-7 and would distort distances, masking the property
        # under test.
        rotation = o3.rand_matrix(dtype=torch.float64)
        rotated_energy, rotated_forces, _, _ = _run(model, POSITIONS @ rotation.T)
        np.testing.assert_allclose(float(rotated_energy), float(energy), rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            rotated_forces.detach().numpy(),
            (forces @ rotation.T).detach().numpy(),
            rtol=0,
            atol=1e-9,
        )


# --------------------------------------------------------------------------
# D2: the attention temperature must reach exactly 1 and never depend on data
# --------------------------------------------------------------------------

def test_temperature_schedule_takes_only_the_epoch():
    source = inspect.getsource(train_module.main)
    body = source.split("def _temperature_scale")[1].split("return")[0]
    assert "force_ema" not in body


@pytest.mark.parametrize(
    "needle",
    [
        "temperature_scale_end must be 1.0",
        "is no longer supported",
        "must be smaller than",
    ],
)
def test_temperature_misconfiguration_is_rejected(needle):
    assert needle in inspect.getsource(train_module.main)


# --------------------------------------------------------------------------
# D3: the energy metric must expose an offset-free error
# --------------------------------------------------------------------------

def test_offset_free_energy_error_is_invariant_to_a_constant_shift():
    """A pure shift of the energy zero must not change the offset-free error."""

    def _collect(shift_ev_per_atom):
        tracker = MetricTracker()
        rng = np.random.default_rng(0)
        for _ in range(25):
            n_atoms = 40
            # Residual-scale energies, as the trainer now passes them: the
            # composition baseline is removed before the comparison.
            target = float(rng.normal(0.0, 1.0))
            scatter = float(rng.normal(0.0, 0.002)) * n_atoms
            predicted = target + scatter + shift_ev_per_atom * n_atoms
            tracker.update(
                torch.tensor(predicted, dtype=torch.float64),
                torch.zeros((n_atoms, 3)),
                torch.zeros((3, 3)),
                torch.tensor(target, dtype=torch.float64),
                torch.zeros((n_atoms, 3)),
                torch.zeros((3, 3)),
                False,
                n_atoms,
            )
        rmse, _, _, _, _, rmse_free, bias = tracker.get_metrics()
        return rmse, rmse_free, bias

    rmse_0, free_0, bias_0 = _collect(0.0)
    rmse_1, free_1, bias_1 = _collect(0.05)

    # A 50 meV/atom shift must move the raw RMSE but not the offset-free error.
    assert rmse_1 > rmse_0 + 10.0
    np.testing.assert_allclose(free_1, free_0, rtol=1e-9)
    np.testing.assert_allclose(bias_1 - bias_0, 50.0, rtol=1e-6)


# --------------------------------------------------------------------------
# D4: the two Sobolev passes must share one dropout realisation
# --------------------------------------------------------------------------

def test_rng_replay_reproduces_an_identical_dropout_forward_pass():
    model = _model(dropout=0.3, seed=1)
    model.train()
    data = {
        "z": torch.tensor(NUMBERS),
        "pos": POSITIONS.clone(),
        "edge_index": _edges(POSITIONS),
        "volume": torch.tensor(1000.0, dtype=torch.float64),
    }

    state = torch.get_rng_state()
    first = float(model(data, training=True, compute_stress=False)[0])
    torch.set_rng_state(state)
    second = float(model(data, training=True, compute_stress=False)[0])
    assert first == second

    # ...and dropout really is active, so the test above is not vacuous.
    third = float(model(data, training=True, compute_stress=False)[0])
    assert third != first


def test_training_loop_replays_the_rng_state_for_the_sobolev_pass():
    source = inspect.getsource(train_module.main)
    assert "rng_state_cpu = torch.get_rng_state()" in source
    assert "torch.set_rng_state(rng_state_cpu)" in source
    # The replay must happen before the perturbed forward, not after.
    replay = source.index("torch.set_rng_state(rng_state_cpu)")
    perturbed = source.index("p_E_pert, _, _, _ = model(")
    assert replay < perturbed


# --------------------------------------------------------------------------
# CUDA readiness: determinism is opt-in and actually configured
# --------------------------------------------------------------------------

def test_determinism_is_off_by_default_and_configurable():
    from train import configure_determinism
    assert configure_determinism({}) is False
    try:
        assert configure_determinism({"deterministic": True}) is True
        import os
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


def test_no_dtype_less_tensor_creation_in_strain_paths():
    """Regression for D7: every strain/stress tensor must inherit pos.dtype."""
    import inspect
    import flashace.model as M
    src = inspect.getsource(M)
    for bad in (
        "torch.zeros(6, device=pos.device, requires_grad=True)",
        "torch.zeros(3, 3, device=pos.device)",
        "torch.eye(3, device=pos.device) + epsilon",
    ):
        assert bad not in src, f"dtype-less tensor creation reintroduced: {bad}"
