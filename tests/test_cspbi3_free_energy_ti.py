import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TI_DIRECTORY = Path(__file__).resolve().parent / "cspbi3" / "free_energy_ti"
sys.path.insert(0, str(TI_DIRECTORY))

from ti_core import (  # noqa: E402
    ANGSTROM_M,
    ATOMIC_MASS_KG,
    BOLTZMANN_J_K,
    ELECTRON_VOLT_J,
    KB_EV_K,
    PLANCK_J_S,
    block_means,
    crossing_temperature,
    einstein_free_energy_per_atom,
    gibbs_helmholtz_curve,
    isobaric_enthalpy_eV,
    pressure_volume_energy_eV,
    symmetric_switching_estimate,
)


class FreeEnergyEquationTests(unittest.TestCase):
    def test_einstein_free_energy_matches_direct_partition_function(self):
        temperature = 450.0
        volume_a3 = 12000.0
        masses_amu = np.asarray([132.905, 207.2, 126.904] * 4)
        springs_ev_a2 = np.asarray([1.2, 3.5, 2.1] * 4)

        beta = 1.0 / (BOLTZMANN_J_K * temperature)
        masses_kg = masses_amu * ATOMIC_MASS_KG
        springs_si = springs_ev_a2 * ELECTRON_VOLT_J / ANGSTROM_M**2
        one_particle_z = (
            beta**2 * springs_si * PLANCK_J_S**2 / (4.0 * math.pi**2 * masses_kg)
        ) ** 1.5
        f_uncorrected = KB_EV_K * temperature * np.sum(np.log(one_particle_z))
        mass_fraction = masses_kg / np.sum(masses_kg)
        compliance = np.sum(mass_fraction**2 / springs_si)
        z_com = volume_a3 * ANGSTROM_M**3 * (beta / (2.0 * math.pi * compliance)) ** 1.5
        expected = (f_uncorrected - KB_EV_K * temperature * math.log(z_com)) / len(masses_amu)

        actual = einstein_free_energy_per_atom(
            temperature, volume_a3, masses_amu, springs_ev_a2
        )
        self.assertAlmostEqual(actual, expected, places=13)

    def test_gibbs_helmholtz_integration_has_correct_sign_and_anchor(self):
        temperature = np.asarray([300.0, 400.0, 500.0])
        coefficient = 2.5e-7
        enthalpy = coefficient * temperature**2
        anchor_g = -3.0
        expected = temperature * (
            anchor_g / 400.0 - coefficient * (temperature - 400.0)
        )
        actual = gibbs_helmholtz_curve(temperature, enthalpy, 400.0, anchor_g)
        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=1.0e-14))
        self.assertAlmostEqual(actual[1], anchor_g)

    def test_forward_reverse_switching_estimator(self):
        coupling_forward = np.linspace(0.0, 1.0, 101)
        d_u_forward = 2.0 + coupling_forward
        coupling_backward = coupling_forward[::-1]
        d_u_backward = 2.0 + coupling_backward
        with tempfile.TemporaryDirectory() as directory:
            forward = Path(directory) / "forward.dat"
            backward = Path(directory) / "backward.dat"
            np.savetxt(
                forward,
                np.column_stack(
                    [np.arange(101), d_u_forward, coupling_forward, np.ones(101) / 100.0]
                ),
                header="step dU lambda dlambda",
            )
            np.savetxt(
                backward,
                np.column_stack(
                    [np.arange(101), d_u_backward, coupling_backward, -np.ones(101) / 100.0]
                ),
                header="step dU lambda dlambda",
            )
            estimate = symmetric_switching_estimate(forward, backward)
        self.assertAlmostEqual(estimate["forward_eV"], 2.5)
        self.assertAlmostEqual(estimate["backward_oriented_eV"], -2.5)
        self.assertAlmostEqual(estimate["reversible_eV"], 2.5)
        self.assertAlmostEqual(estimate["hysteresis_eV"], 0.0, places=13)

    def test_blocking_crossing_and_pressure_volume_conversions(self):
        self.assertTrue(np.allclose(block_means(np.arange(10.0), 4), [3.5, 7.5]))
        self.assertAlmostEqual(crossing_temperature([300.0, 400.0], [-1.0, 3.0]), 325.0)
        self.assertTrue(math.isnan(crossing_temperature([300.0, 400.0], [-1.0, -0.5])))
        expected = 1.0e5 * 1.0e-30 / ELECTRON_VOLT_J
        self.assertAlmostEqual(pressure_volume_energy_eV(1.0, 1.0), expected, places=20)
        enthalpy = isobaric_enthalpy_eV([1.0], [2.0], [1.0], 1.0)
        self.assertAlmostEqual(enthalpy[0], 3.0 + expected, places=15)


if __name__ == "__main__":
    unittest.main()
