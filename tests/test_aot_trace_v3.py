import torch
from torch.fx.experimental.proxy_tensor import make_fx

from flashace.model import TransformersACEV3
from transformers_ace.aot import make_aot_compatible, pad_lammps_inputs
from transformers_ace.deploy import LAMMPSEnergyModel, LAMMPSAOTForceModel


def _inputs():
    return (
        torch.tensor([8, 1, 1, 8], dtype=torch.long),
        torch.tensor(
            [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
        ),
        torch.eye(3) * 9.0,
        torch.tensor(
            [[1, 2, 3, 0, 2, 3, 0, 1], [0, 0, 0, 1, 1, 1, 2, 2]],
            dtype=torch.long,
        ),
        torch.zeros((8, 3)),
        torch.zeros(6),
        torch.ones(4),
    )


def _model(l_max=2):
    torch.manual_seed(17)
    model = TransformersACEV3(
        r_max=4.0,
        l_max=l_max,
        num_radial=3,
        hidden_dim=8,
        num_layers=1,
        correlation_channels=4,
        attention_num_heads=1,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_export_safe_tensor_algebra_matches_e3nn_energy_force_and_virial():
    inputs = _inputs()
    source_model = _model()
    reference = LAMMPSEnergyModel(source_model).eval()
    converted = LAMMPSEnergyModel(make_aot_compatible(source_model)).eval()

    ref_pos = inputs[1].clone().requires_grad_(True)
    ref_strain = inputs[5].clone().requires_grad_(True)
    reference_energy = reference(
        inputs[0], ref_pos, inputs[2], inputs[3], inputs[4], ref_strain, inputs[6]
    )
    reference_force, reference_strain_grad = torch.autograd.grad(
        reference_energy, (ref_pos, ref_strain)
    )

    out_pos = inputs[1].clone().requires_grad_(True)
    out_strain = inputs[5].clone().requires_grad_(True)
    converted_energy = converted(
        inputs[0], out_pos, inputs[2], inputs[3], inputs[4], out_strain, inputs[6]
    )
    converted_force, converted_strain_grad = torch.autograd.grad(
        converted_energy, (out_pos, out_strain)
    )

    torch.testing.assert_close(converted_energy, reference_energy, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(converted_force, reference_force, atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(converted_strain_grad, reference_strain_grad, atol=4e-6, rtol=4e-6)


def test_aot_trace_force_graph_is_static_and_padding_is_physically_null():
    inputs = _inputs()
    program = LAMMPSAOTForceModel(
        LAMMPSEnergyModel(make_aot_compatible(_model())).eval()
    ).eval()
    padded = pad_lammps_inputs(inputs, max_atoms=8, max_edges=16, r_max=4.0)

    direct = program(*inputs)
    padded_output = program(*padded)
    torch.testing.assert_close(padded_output[0], direct[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(padded_output[1][:4], direct[1], atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(padded_output[2], direct[2], atol=4e-6, rtol=4e-6)
    assert padded_output[1][4:].abs().max().item() == 0.0

    graph = make_fx(program)(*padded)
    for node in graph.graph.nodes:
        node.meta.clear()
    graph_output = graph(*padded)
    for actual, expected in zip(graph_output, padded_output):
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)


def test_export_safe_tensor_algebra_supports_all_configured_angular_limits():
    inputs = _inputs()
    for l_max in (0, 1, 2, 3):
        source = _model(l_max=l_max)
        reference = LAMMPSEnergyModel(source).eval()
        converted = LAMMPSEnergyModel(make_aot_compatible(source)).eval()
        expected = reference(*inputs)
        actual = converted(*inputs)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
