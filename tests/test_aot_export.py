"""End-to-end tests for the AOTI export path used by the LAMMPS pair style.

The export pipeline has three places where a mistake is silent rather than
loud: the package can be built without the entries its loader needs, the
compiled program can disagree numerically with eager, and the metadata contract
can misdescribe the model. Each is covered here.

The compile is genuinely slow, so the round-trip runs once on the smallest
model that still exercises the real code path and is shared across tests.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from flashace.model import TransformersACE
from transformers_ace.aot import (
    AOTI_MODEL_NAME,
    compile_aot_force_program,
    load_aot_force_program,
    make_aot_compatible,
    pad_lammps_inputs,
)
from transformers_ace.aot_deploy import _metadata
from transformers_ace.deploy import LAMMPSAOTForceModel, LAMMPSEnergyModel

R_MAX = 4.0
MAX_ATOMS = 16
MAX_EDGES = 96

PAIR_STYLE_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "lammps"
    / "pair_style"
    / "pair_transformers_ace_aot.cpp"
)


def _tiny_program():
    torch.manual_seed(0)
    model = TransformersACE(
        r_max=R_MAX,
        l_max=1,
        num_radial=4,
        hidden_dim=8,
        num_layers=1,
        correlation_order=2,
        correlation_channels=4,
        attention_num_heads=1,
    ).eval()
    energy = LAMMPSEnergyModel(make_aot_compatible(model).eval()).eval()
    return LAMMPSAOTForceModel(energy).eval()


def _tiny_inputs():
    torch.manual_seed(1)
    n_atoms = 6
    positions = torch.rand(n_atoms, 3, dtype=torch.float32) * 3.0
    senders, receivers = [], []
    for receiver in range(n_atoms):
        for sender in range(n_atoms):
            if sender != receiver and torch.linalg.norm(
                positions[sender] - positions[receiver]
            ) < R_MAX:
                senders.append(sender)
                receivers.append(receiver)
    raw = (
        torch.tensor([55, 82, 53, 53, 53, 55]),
        positions,
        torch.eye(3, dtype=torch.float32) * 20.0,
        torch.tensor([senders, receivers], dtype=torch.long),
        torch.zeros((len(senders), 3), dtype=torch.float32),
        torch.zeros(6),
        torch.ones(n_atoms),
    )
    return pad_lammps_inputs(raw, MAX_ATOMS, MAX_EDGES, R_MAX)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    program = _tiny_program()
    inputs = _tiny_inputs()
    reference = program(*inputs)
    package = compile_aot_force_program(
        program, inputs, tmp_path_factory.mktemp("aot") / "tiny.pt2"
    )
    return package, inputs, reference


# --------------------------------------------------------------------------
# The package must be loadable by the loader the pair style actually uses
# --------------------------------------------------------------------------

def test_package_loads_through_the_cpp_loader(exported):
    """Regression: the archive previously lacked the model entry entirely.

    ``package_aoti`` on a bare shared library produced a ``.pt2`` with no
    ``_metadata.json``, so every load failed with "File not found". This is the
    C++ loader the LAMMPS pair style binds, so passing here means the package
    is deployable.
    """
    package, inputs, _ = exported
    assert package.suffix == ".pt2"
    outputs = load_aot_force_program(package)(*inputs)
    assert len(outputs) == 3


def test_package_contains_the_compiled_artifacts(exported):
    """``aot_inductor.package`` must emit the wrapper and its metadata."""
    import zipfile

    package, _, _ = exported
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
    assert any(name.endswith(".wrapper.so") for name in names)
    assert any(name.endswith("_metadata.json") for name in names)
    assert any(f"aotinductor/{AOTI_MODEL_NAME}/" in name for name in names)


# --------------------------------------------------------------------------
# The compiled program must reproduce eager physics
# --------------------------------------------------------------------------

def test_compiled_program_reproduces_eager_energy_forces_and_virial(exported):
    package, inputs, reference = exported
    produced = load_aot_force_program(package)(*inputs)

    energy, forces, virial = reference
    np.testing.assert_allclose(
        float(produced[0]), float(energy), rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        produced[1].numpy(), forces.numpy(), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        produced[2].numpy(), virial.numpy(), rtol=0, atol=1e-4
    )


def test_compiled_forces_stay_zero_on_padded_atoms(exported):
    """Padding must remain inert after compilation, not just in eager."""
    package, inputs, _ = exported
    forces = load_aot_force_program(package)(*inputs)[1]
    assert float(forces[6:].abs().max()) == 0.0


# --------------------------------------------------------------------------
# The metadata contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("version", [2, 3, 4])
def test_metadata_reports_the_checkpoint_architecture(version):
    """Regression: the field was hardcoded to ``trace_v3_or_v4``.

    v2 is the production architecture, so every production export was
    mislabelled.
    """
    text = _metadata(["Cs", "Pb", "I"], 6.0, 256, 4096, version)
    assert f"architecture=trace_v{version}" in text
    assert "trace_v3_or_v4" not in text


def test_metadata_declares_the_format_the_pair_style_requires():
    """The Python writer and the C++ reader must agree on the format string."""
    text = _metadata(["Cs"], 6.0, 16, 64, 2)
    written = re.search(r"^format=(\S+)$", text, re.MULTILINE).group(1)

    source = PAIR_STYLE_SOURCE.read_text()
    expected = re.search(
        r'expected_format\s*=\s*"([^"]+)"', source
    ).group(1)
    assert written == expected


def test_pair_style_rejects_an_unrecognised_metadata_format():
    source = PAIR_STYLE_SOURCE.read_text()
    assert "TRACE AOTI metadata format mismatch" in source


# --------------------------------------------------------------------------
# The Python/C++ ABI contract
# --------------------------------------------------------------------------

def test_cpp_passes_inputs_in_the_order_and_dtype_the_program_expects():
    """The pair style builds the input vector by hand; nothing checks it at runtime.

    AOTI takes a flat tensor vector, so a reordered or retyped argument is not a
    load error -- it is silently wrong dynamics. This pins the C++ call against
    the exported signature and the dtypes ``pad_lammps_inputs`` produces.
    """
    import inspect

    expected_order = list(
        inspect.signature(LAMMPSAOTForceModel.forward).parameters
    )[1:]
    assert expected_order == [
        "z",
        "pos",
        "cell",
        "edge_index",
        "edge_shift",
        "strain",
        "local_mask",
    ]

    source = PAIR_STYLE_SOURCE.read_text()
    vector = re.search(
        r"std::vector<at::Tensor>\s+inputs\s*=\s*\{([^}]*)\}", source
    ).group(1)
    cpp_order = [name.strip() for name in vector.split(",") if name.strip()]
    assert cpp_order == [
        "z_t",
        "pos_t",
        "cell_t",
        "edge_t",
        "device_shift_",
        "device_strain_",
        "mask_t",
    ]

    # Every tensor the C++ builds must carry the dtype the export was traced
    # with; AOTI will not coerce. Per-step buffers come from from_blob; the two
    # structurally constant ones are allocated on the device in
    # allocate_static_inputs and never rewritten.
    dtype_of = dict(
        re.findall(r"auto (\w+_t) = torch::from_blob\([^,]+, \{[^}]*\}, (\w+)\)", source)
    )
    static_block = re.search(
        r"allocate_static_inputs\(\)\s*\{.*?\n\}", source, re.DOTALL
    ).group(0)
    assert "torch::kFloat32" in static_block
    for name in ("device_shift_", "device_strain_"):
        assert f"{name} = torch::zeros(" in static_block
        dtype_of[name] = "f32"
    dtype_of["cell_t"] = "f32"
    assert "kFloat32" in re.search(
        r"cell_tensor\(\) const\s*\{.*?torch::kFloat32", source, re.DOTALL
    ).group(0)

    torch_dtype = {"i64": torch.int64, "f32": torch.float32}
    python_dtypes = [tensor.dtype for tensor in _tiny_inputs()]
    for cpp_name, expected_dtype in zip(cpp_order, python_dtypes):
        assert torch_dtype[dtype_of[cpp_name]] == expected_dtype, (
            f"{cpp_name} is {dtype_of[cpp_name]} in C++ but "
            f"{expected_dtype} in the exported program"
        )


def test_structurally_constant_inputs_are_not_resent_each_step():
    """edge_shift and strain are zero for the whole run; sending them is waste.

    LAMMPS ghosts carry unwrapped coordinates, so the lattice shift is
    identically zero, and the strain derivative is baked into the compiled
    program and evaluated at zero. At max_edges=131072 the shift buffer was
    1.57 MB -- 41% of the per-step host-to-device traffic -- re-zeroed and
    re-sent every timestep to no effect.
    """
    source = PAIR_STYLE_SOURCE.read_text()
    compute = source[source.index("void PairTransformersACEAOT::compute") :]

    # Neither may be rebuilt, refilled or re-copied inside compute().
    assert "host_shift_" not in source
    assert "torch::zeros({6}" not in compute
    assert "device_shift_ =" not in compute
    assert "device_strain_ =" not in compute


def test_per_atom_energy_and_virial_are_refused_rather_than_reported_as_zero():
    """ev_init zeroes eatom/vatom; a pair style that never fills them reports zeros.

    The model returns one extensive energy and a global virial, so no per-atom
    decomposition exists. Silently reporting zeros to compute pe/atom would be
    worse than refusing.
    """
    source = PAIR_STYLE_SOURCE.read_text()
    compute = source[source.index("void PairTransformersACEAOT::compute") :]
    for flag in ("eflag_atom", "vflag_atom"):
        assert f"if ({flag})" in compute, f"{flag} is not guarded"
        # The guard must precede the model call, not trail it.
        assert compute.index(f"if ({flag})") < compute.index("loader_->run")


# --------------------------------------------------------------------------
# The TorchScript export must honour its target device
# --------------------------------------------------------------------------

def test_torchscript_export_traces_on_the_requested_device():
    """Regression: `--device` was ignored and every artifact was CPU-only.

    export_lammps_model forced `model.cpu()` and built CPU example tensors, so
    torch.jit.trace evaluated `pos.device` once and baked a CPU constant into
    `torch.eye(3, dtype=pos.dtype, device=pos.device)`. The saved module loaded
    fine on CUDA and then died on the first step with

        RuntimeError: mat2 is on cpu, different from other tensors on cuda:0

    The device is only observable at trace time, so this pins the source: the
    model and the example inputs must both be moved to the target.
    """
    import inspect
    from transformers_ace.deploy import export_lammps_model

    # Strip comments: the fix is documented in a comment that names the old
    # call, so a naive substring search matches the explanation, not the code.
    source = inspect.getsource(export_lammps_model)
    code = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines()
    )
    assert "model.cpu()" not in code, "export forces the model onto CPU again"
    assert "model.to(target)" in code
    # The example inputs must be moved too; the baked constant comes from the
    # *input* device, not the model's.
    assert "_example_tensors" in code
    moved = re.search(
        r"example_inputs\s*=\s*tuple\(\s*tensor\.to\(target\)", code
    )
    assert moved, "example inputs are not moved to the target device"


def test_torchscript_export_verifies_the_saved_module():
    """check_trace=False removes torch's own check, so the export must verify.

    Without this the only thing standing between a broken trace and an MD run
    is the pair style, which has no way to detect wrong physics.
    """
    import inspect
    from transformers_ace.deploy import export_lammps_model, _verify_traced_model

    assert "_verify_traced_model" in inspect.getsource(export_lammps_model)
    verify = inspect.getsource(_verify_traced_model)
    # It must reload from disk rather than reuse the in-memory module, and do so
    # on the target device.
    assert "torch.jit.load" in verify
    assert "map_location=target" in verify
    assert "raise RuntimeError" in verify
