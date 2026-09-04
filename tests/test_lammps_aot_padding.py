"""The AOTI pair style pads in C++; this pins that it matches the Python contract.

``pad_lammps_inputs`` is the reference. ``_cpp_padding_reference`` reimplements,
line for line, what ``pair_transformers_ace_aot.cpp::compute`` writes into its
staging buffers. If the C++ is edited and drifts from the Python, these tests
fail rather than silently producing wrong forces in MD.
"""

import numpy as np
import pytest
import torch
from ase.io import read
from ase.neighborlist import neighbor_list

from flashace.model import TransformersACE
from transformers_ace.aot import pad_lammps_inputs
from transformers_ace.deploy import LAMMPSEnergyModel

R_MAX = 6.0


def _cpp_padding_reference(z, pos, local_mask, edge_index, max_atoms, max_edges, r_max):
    """Mirror of the C++ staging-buffer fill in pair_transformers_ace_aot.cpp."""
    nall = int(z.shape[0])
    n_edges = int(edge_index.shape[1])

    host_z = torch.ones(max_atoms, dtype=torch.int64)
    host_pos = torch.zeros((max_atoms, 3), dtype=torch.float64)
    host_mask = torch.zeros(max_atoms, dtype=torch.float64)
    host_z[:nall] = z
    host_pos[:nall] = pos
    host_mask[:nall] = local_mask
    host_pos[max_atoms - 1, 0] = 2.0 * r_max

    host_edges = torch.empty((2, max_edges), dtype=torch.int64)
    host_edges[0, :n_edges] = edge_index[0]
    host_edges[1, :n_edges] = edge_index[1]
    host_edges[0, n_edges:] = max_atoms - 1
    host_edges[1, n_edges:] = max_atoms - 2
    host_shift = torch.zeros((max_edges, 3), dtype=torch.float64)
    return host_z, host_pos, host_mask, host_edges, host_shift


def _system(rep=(2, 2, 1)):
    at = read("tests/cspbi3/results/relaxed/cubic_alpha_phase.vasp").repeat(rep)
    i, j, S = neighbor_list("ijS", at, R_MAX)
    z = torch.tensor(at.numbers)
    pos = torch.tensor(at.positions, dtype=torch.float64)
    cell = torch.tensor(at.cell.array, dtype=torch.float64)
    edge_index = torch.stack([torch.tensor(j), torch.tensor(i)])
    edge_shift = torch.zeros((edge_index.shape[1], 3), dtype=torch.float64)
    mask = torch.ones(len(at), dtype=torch.float64)
    return (z, pos, cell, edge_index, edge_shift,
            torch.zeros(6, dtype=torch.float64), mask)


@pytest.mark.parametrize("max_atoms,max_edges", [(128, 2048), (256, 4096), (512, 8192)])
def test_cpp_padding_matches_python_contract(max_atoms, max_edges):
    z, pos, cell, ei, es, strain, mask = _system()
    assert max_atoms >= z.shape[0] + 2 and max_edges >= ei.shape[1]

    ref = pad_lammps_inputs((z, pos, cell, ei, es, strain, mask), max_atoms, max_edges, R_MAX)
    pz, pp, _, pe, ps, _, pm = ref
    cz, cp, cm, ce, cs = _cpp_padding_reference(z, pos, mask, ei, max_atoms, max_edges, R_MAX)

    assert torch.equal(cz, pz), "padded species differ"
    assert torch.equal(cp, pp), "padded positions differ"
    assert torch.equal(cm, pm), "padded local mask differs"
    assert torch.equal(ce, pe), "padded edge index differs"
    assert torch.equal(cs, ps), "padded edge shifts differ"


def test_padded_edges_lie_beyond_the_cutoff():
    """The inertness of padding rests on f_c(2 r_c) == 0 exactly."""
    from flashace.physics import SmoothPolynomialCutoff

    z, pos, cell, ei, es, strain, mask = _system()
    pz, pp, _, pe, _, _, _ = pad_lammps_inputs(
        (z, pos, cell, ei, es, strain, mask), 256, 4096, R_MAX
    )
    n_edges = ei.shape[1]
    d = (pp[pe[0, n_edges:]] - pp[pe[1, n_edges:]]).norm(dim=1)
    assert float(d.min()) >= R_MAX, f"a padded edge is inside the cutoff: {float(d.min())}"
    cut = SmoothPolynomialCutoff(R_MAX).double()
    assert float(cut(d).abs().max()) == 0.0, "padded edges have nonzero cutoff weight"


def test_padding_is_energetically_inert_end_to_end():
    """Energy, forces and virial must be bitwise identical with and without padding."""
    z, pos, cell, ei, es, strain, mask = _system()
    ck = torch.load("training/h100_run/model_h100.pt", map_location="cpu")
    cfg = ck["config"]
    keys = ["r_max", "l_max", "num_radial", "hidden_dim", "num_layers", "correlation_order",
            "correlation_channels", "radial_mlp_hidden", "radial_mlp_layers",
            "attention_num_heads", "attention_key_dim", "attention_ffn_hidden",
            "attention_dropout", "attention_layer_scale_init", "attention_distance_penalty",
            "radial_basis_type", "radial_trainable", "gaussian_width"]
    m = TransformersACE(**{k: cfg[k] for k in keys if k in cfg})
    m.load_state_dict(ck["model_state_dict"])
    dep = LAMMPSEnergyModel(m.double().eval()).double().eval()

    def evaluate(inp):
        z_, p_, c_, e_, s_, st_, mk_ = inp
        p_ = p_.clone().requires_grad_(True)
        st_ = st_.clone().requires_grad_(True)
        E = dep(z_, p_, c_, e_, s_, st_, mk_)
        gp, gs = torch.autograd.grad(E, (p_, st_))
        return float(E), -gp, gs

    n = int(z.shape[0])
    E0, F0, V0 = evaluate((z, pos, cell, ei, es, strain, mask))
    for ma, me in ((n + 2, ei.shape[1]), (n + 64, ei.shape[1] + 2048)):
        E1, F1, V1 = evaluate(pad_lammps_inputs((z, pos, cell, ei, es, strain, mask), ma, me, R_MAX))
        assert E1 == E0, f"energy changed by {abs(E1 - E0)}"
        assert torch.equal(F1[:n], F0), "real forces changed"
        assert float(F1[n:].abs().max()) == 0.0, "padding atoms carry force"
        assert torch.equal(V1, V0), "virial changed"
