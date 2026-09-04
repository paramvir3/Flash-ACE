import numpy as np
import torch

from flashace.model import TransformersACE
from transformers_ace.deploy import LAMMPSEnergyModel


def _periodic_data():
    cell = torch.tensor(
        [
            [5.3, 0.0, 0.0],
            [0.4, 5.7, 0.0],
            [-0.2, 0.3, 6.1],
        ],
        dtype=torch.float32,
    )
    pos = torch.tensor(
        [
            [0.2, 0.4, 0.6],
            [1.8, 0.7, 0.9],
            [0.9, 2.0, 1.4],
            [4.7, 5.2, 5.5],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [1, 2, 3, 0, 2, 3, 0, 1],
            [0, 0, 0, 1, 1, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    edge_shift = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    z = torch.tensor([8, 1, 1, 8], dtype=torch.long)
    return {
        "z": z,
        "pos": pos,
        "cell": cell,
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "volume": torch.abs(torch.det(cell)),
    }


def test_deployed_strain_virial_matches_python_stress():
    torch.manual_seed(11)
    model = TransformersACE(
        r_max=4.0,
        l_max=1,
        num_radial=4,
        hidden_dim=8,
        num_layers=1,
        correlation_channels=4,
        attention_num_heads=1,
    ).eval()

    data = _periodic_data()
    energy, forces, stress, _ = model(data, training=False, compute_stress=True)
    volume = data["volume"]

    deploy_model = LAMMPSEnergyModel(model).eval()
    pos = data["pos"].clone().requires_grad_(True)
    strain = torch.zeros(6, dtype=torch.float32, requires_grad=True)
    local_mask = torch.ones(len(data["z"]), dtype=torch.float32)
    deployed_energy = deploy_model(
        data["z"],
        pos,
        data["cell"],
        data["edge_index"],
        data["edge_shift"],
        strain,
        local_mask,
    )
    grad_pos, grad_strain = torch.autograd.grad(deployed_energy, (pos, strain))
    deployed_forces = -grad_pos
    deployed_virial_lammps_order = torch.stack(
        (
            -grad_strain[0],
            -grad_strain[1],
            -grad_strain[2],
            -0.5 * grad_strain[3],
            -0.5 * grad_strain[4],
            -0.5 * grad_strain[5],
        )
    )
    stress_virial_lammps_order = -volume * torch.stack(
        (
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[0, 1],
            stress[0, 2],
            stress[1, 2],
        )
    )

    np.testing.assert_allclose(
        deployed_energy.detach().numpy(),
        energy.detach().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        deployed_forces.detach().numpy(),
        forces.detach().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        deployed_virial_lammps_order.detach().numpy(),
        stress_virial_lammps_order.detach().numpy(),
        rtol=2e-5,
        atol=2e-6,
    )
