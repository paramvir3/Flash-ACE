import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from train import AtomisticDataset


def test_nonperiodic_molecule_does_not_require_a_cell_volume():
    atoms = Atoms(
        "HCN",
        positions=[
            [0.0, 0.0, 0.0],
            [1.06, 0.0, 0.0],
            [2.22, 0.0, 0.0],
        ],
        pbc=False,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-93.0,
        forces=np.zeros((3, 3)),
    )

    item = AtomisticDataset([atoms], r_max=4.0)[0]

    assert atoms.cell.rank == 0
    assert item["volume"].item() == 1.0
    assert not item["has_stress"].item()
    assert item["edge_index"].shape == (2, 6)


def test_periodic_structure_keeps_physical_cell_volume():
    atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[2.0, 3.0, 4.0],
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.0,
        forces=np.zeros((1, 3)),
    )

    item = AtomisticDataset([atoms], r_max=1.0)[0]

    assert item["volume"].item() == 24.0
