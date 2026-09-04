"""Thermodynamic equations used by the CsPbI3 free-energy workflow."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

# CODATA values.  k_B, h, and the electron volt are exact in the revised SI.
BOLTZMANN_J_K = 1.380649e-23
PLANCK_J_S = 6.62607015e-34
ELECTRON_VOLT_J = 1.602176634e-19
ATOMIC_MASS_KG = 1.66053906660e-27
ANGSTROM_M = 1.0e-10
BAR_PA = 1.0e5
KB_EV_K = BOLTZMANN_J_K / ELECTRON_VOLT_J
EV_PER_FORMULA_TO_KJ_MOL = 96.48533212331002


def einstein_free_energy_per_atom(
    temperature_K: float,
    volume_A3: float,
    masses_amu: Sequence[float],
    springs_eV_A2: Sequence[float],
) -> float:
    """Return the classical Einstein-crystal Helmholtz free energy in eV/atom.

    The center-of-mass correction is the one appropriate to a three-dimensional
    periodic solid whose center of mass is constrained during Frenkel--Ladd
    integration.  Species-dependent spring constants are supported.
    """

    temperature_K = float(temperature_K)
    volume_A3 = float(volume_A3)
    masses_amu = np.asarray(masses_amu, dtype=float)
    springs_eV_A2 = np.asarray(springs_eV_A2, dtype=float)

    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if volume_A3 <= 0.0:
        raise ValueError("volume_A3 must be positive")
    if masses_amu.ndim != 1 or springs_eV_A2.ndim != 1:
        raise ValueError("masses and springs must be one-dimensional")
    if len(masses_amu) == 0 or len(masses_amu) != len(springs_eV_A2):
        raise ValueError("masses and springs must have the same nonzero length")
    if np.any(masses_amu <= 0.0) or np.any(springs_eV_A2 <= 0.0):
        raise ValueError("all masses and spring constants must be positive")

    beta_J_inv = 1.0 / (BOLTZMANN_J_K * temperature_K)
    masses_kg = masses_amu * ATOMIC_MASS_KG
    springs_J_m2 = springs_eV_A2 * ELECTRON_VOLT_J / (ANGSTROM_M**2)

    # One-particle classical harmonic partition factor, including momentum.
    z_einstein = (
        beta_J_inv**2
        * springs_J_m2
        * PLANCK_J_S**2
        / (4.0 * math.pi**2 * masses_kg)
    ) ** 1.5
    f_uncorrected = KB_EV_K * temperature_K * np.sum(np.log(z_einstein))

    mass_fraction = masses_kg / np.sum(masses_kg)
    weighted_compliance = np.sum(mass_fraction**2 / springs_J_m2)
    volume_m3 = volume_A3 * ANGSTROM_M**3
    com_partition_factor = volume_m3 * (
        beta_J_inv / (2.0 * math.pi * weighted_compliance)
    ) ** 1.5
    f_com = KB_EV_K * temperature_K * math.log(com_partition_factor)

    return float((f_uncorrected - f_com) / len(masses_amu))


def pressure_volume_energy_eV(pressure_bar: float, volume_A3: float) -> float:
    """Convert P V from bar Angstrom^3 to eV."""

    return float(pressure_bar) * BAR_PA * float(volume_A3) * ANGSTROM_M**3 / ELECTRON_VOLT_J


def isobaric_enthalpy_eV(
    potential_energy_eV: Sequence[float],
    kinetic_energy_eV: Sequence[float],
    volume_A3: Sequence[float],
    external_pressure_bar: float,
) -> np.ndarray:
    """Return H=U+K+P_external V for isothermal--isobaric samples."""

    potential = np.asarray(potential_energy_eV, dtype=float)
    kinetic = np.asarray(kinetic_energy_eV, dtype=float)
    volume = np.asarray(volume_A3, dtype=float)
    if potential.shape != kinetic.shape or potential.shape != volume.shape:
        raise ValueError("potential energy, kinetic energy, and volume shapes differ")
    pv = float(external_pressure_bar) * BAR_PA * volume * ANGSTROM_M**3 / ELECTRON_VOLT_J
    return potential + kinetic + pv


def load_switching_table(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load lambda and U_real-U_Einstein from a LAMMPS switching file."""

    table = np.loadtxt(str(path), comments="#", ndmin=2)
    if table.shape[1] < 4:
        raise ValueError("{} must contain step, dU, lambda, and dlambda".format(path))
    d_u = np.asarray(table[:, 1], dtype=float)
    coupling = np.asarray(table[:, 2], dtype=float)
    if len(coupling) < 2 or not np.all(np.isfinite(d_u)) or not np.all(np.isfinite(coupling)):
        raise ValueError("{} does not contain a finite switching trajectory".format(path))
    return coupling, d_u


def switching_integral(path: Path) -> float:
    """Integrate (U_real-U_Einstein) d lambda in eV."""

    coupling, d_u = load_switching_table(path)
    delta_coupling = np.diff(coupling)
    # Keep signed d lambda: the reverse trajectory must integrate from 1 to 0.
    return float(np.sum(0.5 * (d_u[:-1] + d_u[1:]) * delta_coupling))


def symmetric_switching_estimate(forward_path: Path, backward_path: Path) -> Dict[str, float]:
    """Combine forward and reverse work to cancel leading dissipative error."""

    forward = switching_integral(forward_path)
    backward_oriented = switching_integral(backward_path)
    return {
        "forward_eV": forward,
        "backward_oriented_eV": backward_oriented,
        "reversible_eV": 0.5 * (forward - backward_oriented),
        "hysteresis_eV": 0.5 * (forward + backward_oriented),
    }


def gibbs_helmholtz_curve(
    temperatures_K: Sequence[float],
    enthalpies_eV_atom: Sequence[float],
    anchor_temperature_K: float,
    anchor_gibbs_eV_atom: float,
) -> np.ndarray:
    r"""Propagate G along an isobar using d(G/T)/dT = -H/T^2."""

    temperatures = np.asarray(temperatures_K, dtype=float)
    enthalpies = np.asarray(enthalpies_eV_atom, dtype=float)
    if temperatures.ndim != 1 or len(temperatures) != len(enthalpies):
        raise ValueError("temperatures and enthalpies must be equal-length vectors")
    if len(temperatures) == 0 or np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")
    if np.any(np.diff(temperatures) <= 0.0):
        raise ValueError("temperatures must be strictly increasing")

    anchor_indices = np.flatnonzero(np.isclose(temperatures, anchor_temperature_K, atol=1.0e-10))
    if len(anchor_indices) != 1:
        raise ValueError("anchor temperature must occur exactly once in the temperature grid")
    anchor_index = int(anchor_indices[0])

    integrand = enthalpies / temperatures**2
    integral = np.zeros_like(temperatures)
    for index in range(anchor_index + 1, len(temperatures)):
        delta_t = temperatures[index] - temperatures[index - 1]
        integral[index] = integral[index - 1] + 0.5 * delta_t * (
            integrand[index - 1] + integrand[index]
        )
    for index in range(anchor_index - 1, -1, -1):
        delta_t = temperatures[index + 1] - temperatures[index]
        integral[index] = integral[index + 1] - 0.5 * delta_t * (
            integrand[index + 1] + integrand[index]
        )

    return temperatures * (float(anchor_gibbs_eV_atom) / anchor_temperature_K - integral)


def block_means(values: Sequence[float], samples_per_block: int) -> np.ndarray:
    """Return nonoverlapping block means, discarding only a leading remainder."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a nonempty vector")
    if samples_per_block <= 0:
        raise ValueError("samples_per_block must be positive")
    if samples_per_block > len(array):
        return np.asarray([np.mean(array)], dtype=float)
    n_blocks = len(array) // samples_per_block
    trimmed = array[-n_blocks * samples_per_block :]
    return np.mean(trimmed.reshape(n_blocks, samples_per_block), axis=1)


def crossing_temperature(temperatures_K: Sequence[float], delta_g: Sequence[float]) -> float:
    """Find the first linear-interpolated zero crossing; return NaN if absent."""

    temperature = np.asarray(temperatures_K, dtype=float)
    values = np.asarray(delta_g, dtype=float)
    if len(temperature) != len(values):
        raise ValueError("temperature and delta_g lengths differ")
    exact = np.flatnonzero(np.isclose(values, 0.0, atol=1.0e-14))
    if len(exact):
        return float(temperature[int(exact[0])])
    signs = np.signbit(values)
    changes = np.flatnonzero(signs[:-1] != signs[1:])
    if not len(changes):
        return float("nan")
    index = int(changes[0])
    return float(
        temperature[index]
        - values[index]
        * (temperature[index + 1] - temperature[index])
        / (values[index + 1] - values[index])
    )


def read_named_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read a numeric CSV with a single header row."""

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(line for line in handle if line.strip() and not line.startswith("#"))
        rows = list(reader)
    if not rows:
        raise ValueError("{} contains no data rows".format(path))
    output = {}
    for name in rows[0]:
        output[name.strip()] = np.asarray([float(row[name]) for row in rows], dtype=float)
    return output


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Write deterministic CSV output."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
