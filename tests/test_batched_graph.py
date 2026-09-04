"""Batched-graph evaluation must reproduce per-structure evaluation exactly.

The batched path exists purely as a GPU throughput optimisation, so its only
correctness requirement is that it is indistinguishable from looping over
structures. These tests pin that, including the periodic stress path.
"""

import numpy as np
import pytest
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from e3nn import o3

from flashace.model import TransformersACE
from train import AtomisticDataset

R_MAX = 5.0


def _model(seed=0):
    torch.manual_seed(seed)
    m = TransformersACE(
        r_max=R_MAX, l_max=2, num_radial=6, hidden_dim=16, num_layers=1,
        correlation_order=4, correlation_channels=8, attention_num_heads=2,
        attention_dropout=0.0,
    ).double()
    m.eval()
    return m


def _frames(n=4):
    frames = read("training/train.extxyz", index=f":{n}")
    return frames


def _item(atoms):
    i, j, S = neighbor_list("ijS", atoms, R_MAX)
    return {
        "z": torch.tensor(atoms.numbers),
        "pos": torch.tensor(atoms.positions, dtype=torch.float64),
        "cell": torch.tensor(atoms.cell.array, dtype=torch.float64),
        "volume": torch.tensor(float(atoms.get_volume()), dtype=torch.float64),
        "edge_index": torch.stack([torch.tensor(j), torch.tensor(i)]),
        "edge_shift": torch.tensor(S, dtype=torch.float64),
        "t_E": torch.tensor(0.0, dtype=torch.float64),
        "t_F": torch.zeros((len(atoms), 3), dtype=torch.float64),
        "t_S": torch.zeros((3, 3), dtype=torch.float64),
        "has_stress": torch.tensor(True),
    }


@pytest.mark.parametrize("compute_stress", [False, True])
def test_batched_matches_sequential(compute_stress):
    m = _model()
    frames = _frames(4)
    items = [_item(a) for a in frames]

    seq_E, seq_F, seq_S = [], [], []
    for it in items:
        E, F, S, _ = m(it, training=False, compute_stress=compute_stress)
        seq_E.append(float(E)); seq_F.append(F.detach()); seq_S.append(S.detach())

    bat = AtomisticDataset.collate_batched(items)
    bE, bF, bS, _ = m.forward_batched(bat, training=False, compute_stress=compute_stress)

    np.testing.assert_allclose(bE.detach().numpy(), np.array(seq_E), rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        bF.detach().numpy(), torch.cat(seq_F).numpy(), rtol=0, atol=1e-9
    )
    if compute_stress:
        np.testing.assert_allclose(
            bS.detach().numpy(), torch.stack(seq_S).numpy(), rtol=0, atol=1e-11
        )


def test_batched_graph_is_disconnected():
    """No edge may cross a structure boundary."""
    items = [_item(a) for a in _frames(4)]
    bat = AtomisticDataset.collate_batched(items)
    b = bat["batch"]
    src, dst = bat["edge_index"][0], bat["edge_index"][1]
    assert torch.equal(b[src], b[dst]), "an edge connects two different structures"


def test_batched_forces_are_conservative():
    m = _model(seed=1)
    items = [_item(a) for a in _frames(2)]
    bat = AtomisticDataset.collate_batched(items)
    _, F, _, _ = m.forward_batched(bat, training=False, compute_stress=False)

    h = 1e-4
    pos0 = bat["pos"].clone()
    worst = 0.0
    for k in (0, 5, 11):
        for a in range(3):
            for sign in (+1, -1):
                p = pos0.clone(); p[k, a] += sign * h
                bat2 = dict(bat); bat2["pos"] = p
                E = m.forward_batched(bat2, training=False, compute_stress=False)[0].sum()
                if sign > 0: Ep = float(E)
                else: Em = float(E)
            worst = max(worst, abs(-(Ep - Em) / (2 * h) - float(F[k, a])))
    assert worst < 1e-5, f"max |F - F_FD| = {worst:.3e}"


def test_single_structure_forward_is_unchanged():
    """forward() must be untouched: the deployed paths depend on it."""
    m = _model(seed=2)
    it = _item(_frames(1)[0])
    E1, F1, S1, _ = m(it, training=False, compute_stress=True)
    bat = AtomisticDataset.collate_batched([it])
    E2, F2, S2, _ = m.forward_batched(bat, training=False, compute_stress=True)
    np.testing.assert_allclose(float(E2[0]), float(E1), rtol=0, atol=1e-10)
    np.testing.assert_allclose(F2.detach().numpy(), F1.detach().numpy(), rtol=0, atol=1e-10)
    np.testing.assert_allclose(S2[0].detach().numpy(), S1.detach().numpy(), rtol=0, atol=1e-12)


def test_float64_stress_works_without_a_global_default_dtype():
    """Regression for D7: the strain path used dtype-less torch.eye/zeros.

    A float64 model whose stress path silently created float32 strain tensors
    raised 'expected m1 and m2 to have the same dtype' unless the process-wide
    default dtype happened to be float64.
    """
    assert torch.get_default_dtype() == torch.float32, "test must run with the fp32 default"
    m = _model(seed=3)  # .double() model, global default still fp32
    it = _item(_frames(1)[0])
    E, F, S, _ = m(it, training=False, compute_stress=True)
    assert E.dtype == torch.float64 and S.dtype == torch.float64
    bat = AtomisticDataset.collate_batched([it])
    Eb, Fb, Sb, _ = m.forward_batched(bat, training=False, compute_stress=True)
    assert Sb.dtype == torch.float64


def _seq_loss_and_grads(model, items, w_E, w_F, w_S):
    """Reference: exactly what the per-structure training loop computes."""
    model.zero_grad(set_to_none=True)
    total = 0.0
    for it in items:
        E, F, S, _ = model(it, training=True, compute_stress=True)
        n = int(it["z"].shape[0])
        loss_e = ((E - it["t_E"]) / n) ** 2
        loss_f = torch.mean((F - it["t_F"]) ** 2)
        from train import stress_to_voigt
        loss_s = torch.mean((stress_to_voigt(S) - stress_to_voigt(it["t_S"])) ** 2)
        li = w_E * loss_e + w_F * loss_f + w_S * loss_s
        (li / len(items)).backward()
        total += float(li.detach())
    g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return total, g.clone()


def _bat_loss_and_grads(model, items, w_E, w_F, w_S):
    from train import AtomisticDataset as DS
    model.zero_grad(set_to_none=True)
    bat = DS.collate_batched(items)
    n_at = bat["n_atoms"].to(bat["pos"].dtype)
    E, F, S, _ = model.forward_batched(bat, training=True, compute_stress=True)
    loss_e = (((E - bat["t_E"]) / n_at) ** 2).sum()
    dF2 = (F - bat["t_F"]) ** 2
    per_f = torch.zeros_like(E).index_add(0, bat["batch"], dF2.sum(dim=1))
    loss_f = (per_f / (3.0 * n_at)).sum()
    dS2 = ((S - bat["t_S"]) ** 2).flatten(1)
    voigt = dS2[:, [0, 4, 8, 5, 2, 1]].sum(dim=1)
    loss_s = (voigt / 6.0).sum()
    loss = w_E * loss_e + w_F * loss_f + w_S * loss_s
    (loss / len(items)).backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return float(loss.detach()), g.clone()


def test_batched_training_step_matches_sequential_loss_and_gradients():
    """The batched loss must equal Eqs. (37)-(40) evaluated structure by structure,
    and must produce the same parameter gradients."""
    m = _model(seed=4)
    items = [_item(a) for a in _frames(4)]
    rng = torch.Generator().manual_seed(0)
    for it in items:                      # non-trivial targets
        it["t_E"] = torch.randn((), generator=rng, dtype=torch.float64)
        it["t_F"] = torch.randn(it["t_F"].shape, generator=rng, dtype=torch.float64) * 0.1
        it["t_S"] = torch.randn((3, 3), generator=rng, dtype=torch.float64) * 1e-3
        it["t_S"] = 0.5 * (it["t_S"] + it["t_S"].T)

    w_E, w_F, w_S = 1.0, 10.0, 1000.0
    l_seq, g_seq = _seq_loss_and_grads(m, items, w_E, w_F, w_S)
    l_bat, g_bat = _bat_loss_and_grads(m, items, w_E, w_F, w_S)

    assert abs(l_bat - l_seq) / abs(l_seq) < 1e-10, f"loss {l_bat} vs {l_seq}"
    rel = float((g_bat - g_seq).norm() / g_seq.norm())
    assert rel < 1e-9, f"gradient relative difference {rel:.3e}"


def test_update_batched_matches_per_structure_metrics():
    """The on-device batched tracker must give identical metrics to the
    per-structure one. It exists only to remove host synchronisation."""
    from train import MetricTracker

    rng = torch.Generator().manual_seed(7)
    B = 5
    n_atoms = torch.tensor([40, 40, 40, 40, 40])
    batch = torch.repeat_interleave(torch.arange(B), n_atoms)
    N = int(n_atoms.sum())

    p_E = torch.randn(B, generator=rng, dtype=torch.float64) * 2.0
    t_E = torch.randn(B, generator=rng, dtype=torch.float64) * 2.0
    p_F = torch.randn(N, 3, generator=rng, dtype=torch.float64)
    t_F = torch.randn(N, 3, generator=rng, dtype=torch.float64)
    p_S = torch.randn(B, 3, 3, generator=rng, dtype=torch.float64) * 1e-3
    t_S = torch.randn(B, 3, 3, generator=rng, dtype=torch.float64) * 1e-3
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

    seq = MetricTracker()
    for b in range(B):
        sel = batch == b
        seq.update(
            p_E[b], p_F[sel], p_S[b], t_E[b], t_F[sel], t_S[b],
            bool(mask[b] > 0), int(n_atoms[b]),
        )
    bat = MetricTracker()
    bat.update_batched(p_E, p_F, p_S, t_E, t_F, t_S, mask, batch, n_atoms)

    names = ["rmse_e", "force_rmse", "rmse_s", "force_mse", "force_mae",
             "rmse_e_free", "bias"]
    for name, a, b_ in zip(names, seq.get_metrics(), bat.get_metrics()):
        np.testing.assert_allclose(b_, a, rtol=1e-10, atol=1e-12,
                                   err_msg=f"metric {name} differs")


def test_update_batched_offset_free_is_shift_invariant():
    """Chan's parallel merge must keep the offset-free error exact under a
    constant shift, the property the metric exists for."""
    from train import MetricTracker

    rng = torch.Generator().manual_seed(11)
    n_atoms = torch.tensor([40] * 6)
    batch = torch.repeat_interleave(torch.arange(6), n_atoms)
    N = int(n_atoms.sum())
    base = torch.randn(6, generator=rng, dtype=torch.float64) * 0.05
    zeroF = torch.zeros(N, 3, dtype=torch.float64)
    zeroS = torch.zeros(6, 3, 3, dtype=torch.float64)
    out = []
    for shift in (0.0, 2.0):                      # 2 eV per structure = 50 meV/atom
        t = MetricTracker()
        t.update_batched(base + shift, zeroF, zeroS, torch.zeros(6, dtype=torch.float64),
                         zeroF, zeroS, torch.zeros(6, dtype=torch.float64), batch, n_atoms)
        m = t.get_metrics()
        out.append((m[0], m[5], m[6]))            # rmse_e, offset-free, bias
    assert out[1][0] > out[0][0] + 10.0, "raw RMSE should move with the shift"
    np.testing.assert_allclose(out[1][1], out[0][1], rtol=1e-9)
    np.testing.assert_allclose(out[1][2] - out[0][2], 50.0, rtol=1e-6)


def test_freezing_the_layer_scale_changes_nothing_numerically():
    """freeze_layer_scales() bakes a constant; predictions must be bit-identical."""
    from flashace.model import freeze_layer_scales

    m = _model(seed=5)
    with torch.no_grad():                     # non-trivial, anisotropic across copies
        m.layers[0].layer_scale_attn.copy_(
            torch.linspace(-0.7, 1.1, m.layers[0].layer_scale_attn.numel(), dtype=torch.float64)
        )
    it = _item(_frames(1)[0])
    E0, F0, S0, _ = m(it, training=False, compute_stress=True)

    n = freeze_layer_scales(m)
    assert n >= 1, "no blocks were frozen"
    assert m.layers[0]._frozen_layer_scale is not None

    E1, F1, S1, _ = m(it, training=False, compute_stress=True)
    assert float(E1) == float(E0)
    assert torch.equal(F1, F0)
    assert torch.equal(S1, S0)


def test_frozen_layer_scale_is_the_expanded_parameter():
    from flashace.model import freeze_layer_scales, _irrepwise_layer_scale

    m = _model(seed=6)
    freeze_layer_scales(m)
    blk = m.layers[0]
    expected = _irrepwise_layer_scale(blk.layer_scale_attn.detach(), blk._layer_scale_blocks)
    assert torch.equal(blk._frozen_layer_scale, expected)
    assert blk._frozen_layer_scale.numel() == o3.Irreps(blk.node_irreps).dim
