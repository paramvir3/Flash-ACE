#!/usr/bin/env python3
"""Prepare nonperiodic TRACE datasets from the published Figshare NPZ files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write


FIGSHARE_ARTICLE = "https://doi.org/10.6084/m9.figshare.28631591.v4"
DOWNLOAD_ROOT = "https://ndownloader.figshare.com/files"


@dataclass(frozen=True)
class SourceFile:
    file_id: int
    filename: str
    md5: str

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_ROOT}/{self.file_id}"


R2_FILES = {
    "wtmetad": SourceFile(59117249, "r2_wtmetad_al.npz", "f467b7ae9f2f395af500083876ced250"),
    "downhill": SourceFile(59117315, "r2_downhill_al.npz", "65023cab1ad7eb53e26abc11b41952d2"),
    "us_test": SourceFile(59117237, "r2_wtmetad_us_test.npz", "d2149e652b6fc4f7bed972a5d513e512"),
    "reactant": SourceFile(59117264, "r_r2.xyz", "338cad0bc2c6019e9f1755f4167b8d2b"),
    "transition_state": SourceFile(59117267, "ts_r2.xyz", "ee904f5d503d01e03c0994d85d0c2cb5"),
    "product": SourceFile(59117261, "p_r2.xyz", "2288c865a32ffee5a52faa507f27e44c"),
    "irc_coordinates": SourceFile(59117297, "r2_irc.xyz", "4d4a6b56ce4e59fc6d2ee52ab0331c4a"),
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifies the publisher-provided checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(source: SourceFile, cache: Path) -> Path:
    destination = cache / f"{source.file_id}_{source.filename}"
    if not destination.exists() or md5sum(destination) != source.md5:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source.url) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = md5sum(destination)
    if actual != source.md5:
        raise ValueError(
            f"Checksum mismatch for {source.filename}: expected {source.md5}, got {actual}"
        )
    return destination


def r2_coordinate(
    positions: np.ndarray, numbers: np.ndarray
) -> Tuple[float, float, float]:
    if any(int(numbers[index]) != 6 for index in (10, 11, 14)):
        raise ValueError("The published R2 CV indices no longer identify carbon atoms")
    r_1 = float(np.linalg.norm(positions[14] - positions[11]))
    r_2 = float(np.linalg.norm(positions[14] - positions[10]))
    return r_1, r_2, r_1 - r_2


def load_npz_frames(
    path: Path,
    source_name: str,
    coordinate: Callable[[np.ndarray, np.ndarray], Tuple[float, float, float]],
    coordinate_names: Sequence[str],
    energy_reference: float,
    expected_charge: int,
    expected_multiplicity: int,
) -> List[Atoms]:
    archive = np.load(path, allow_pickle=True)
    positions = np.asarray(archive["R"], dtype=float)
    numbers = np.asarray(archive["Z"], dtype=int)
    energies = np.asarray(archive["E_true"], dtype=float)
    forces = np.asarray(archive["F_true"], dtype=float)
    charges = np.asarray(archive["C"], dtype=int)
    multiplicities = np.asarray(archive["M"], dtype=int)

    nframes = len(energies)
    if not (
        len(positions) == len(numbers) == len(forces) == len(charges) == len(multiplicities) == nframes
    ):
        raise ValueError(f"Inconsistent array lengths in {path}")
    if not (np.isfinite(positions).all() and np.isfinite(energies).all() and np.isfinite(forces).all()):
        raise ValueError(f"Non-finite labels in {path}")
    if np.any(charges != expected_charge) or np.any(multiplicities != expected_multiplicity):
        raise ValueError(f"Unexpected charge or multiplicity in {path}")

    frames = []
    reference_numbers = tuple(int(value) for value in numbers[0])
    for index in range(nframes):
        if tuple(int(value) for value in numbers[index]) != reference_numbers:
            raise ValueError(f"Atom ordering changes within {path} at frame {index}")
        atoms = Atoms(numbers=numbers[index], positions=positions[index], pbc=False)
        atoms.info.update(
            source=source_name,
            source_frame=index,
            charge=expected_charge,
            multiplicity=expected_multiplicity,
            absolute_energy_eV=float(energies[index]),
            energy_reference_eV=float(energy_reference),
        )
        for name, value in zip(
            coordinate_names, coordinate(positions[index], numbers[index])
        ):
            atoms.info[name] = value
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(energies[index] - energy_reference),
            forces=forces[index].copy(),
        )
        frames.append(atoms)
    return frames


def geometry_fingerprint(atoms: Atoms, tolerance: float = 1.0e-5) -> str:
    distances = np.linalg.norm(
        atoms.positions[:, None, :] - atoms.positions[None, :, :], axis=-1
    )
    upper = distances[np.triu_indices(len(atoms), 1)]
    quantized = np.rint(upper / tolerance).astype(np.int64)
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.numbers, dtype=np.int16).tobytes())
    digest.update(quantized.tobytes())
    return digest.hexdigest()


def deduplicate(frames: Iterable[Atoms]) -> Tuple[List[Atoms], int]:
    unique: Dict[str, Atoms] = {}
    duplicates = 0
    for atoms in frames:
        fingerprint = geometry_fingerprint(atoms)
        if fingerprint in unique:
            duplicates += 1
            previous = unique[fingerprint]
            if abs(previous.get_potential_energy() - atoms.get_potential_energy()) > 1.0e-5:
                raise ValueError("Geometrically duplicate frames have inconsistent energy labels")
            continue
        unique[fingerprint] = atoms
    return list(unique.values()), duplicates


def stratified_validation_split(
    frames: Sequence[Atoms], coordinate_name: str, fraction: float
) -> Tuple[List[Atoms], List[Atoms]]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation_fraction must lie between 0 and 0.5")
    train_indices = set(range(len(frames)))
    validation_indices = set()
    sources = sorted({str(atoms.info["source"]) for atoms in frames})

    for source in sources:
        indices = [i for i, atoms in enumerate(frames) if atoms.info["source"] == source]
        indices.sort(key=lambda i: float(frames[i].info[coordinate_name]))
        n_validation = max(1, int(round(fraction * len(indices))))
        # Evenly cover the reaction coordinate while retaining its extrema in training.
        locations = np.linspace(0, len(indices) - 1, n_validation + 2)[1:-1]
        selected = {indices[int(round(location))] for location in locations}
        if len(selected) != n_validation:
            for index in indices[1:-1]:
                selected.add(index)
                if len(selected) == n_validation:
                    break
        validation_indices.update(selected)

    train_indices.difference_update(validation_indices)
    train = [frames[index] for index in sorted(train_indices)]
    validation = [frames[index] for index in sorted(validation_indices)]
    if not train or not validation:
        raise ValueError("The deterministic split produced an empty subset")
    return train, validation


def frame_statistics(frames: Sequence[Atoms], coordinate_name: str) -> dict:
    energies = np.asarray([atoms.get_potential_energy() for atoms in frames])
    coordinates = np.asarray([atoms.info[coordinate_name] for atoms in frames])
    maximum_forces = np.asarray(
        [np.linalg.norm(atoms.get_forces(), axis=1).max() for atoms in frames]
    )
    net_forces = np.asarray([np.linalg.norm(atoms.get_forces().sum(axis=0)) for atoms in frames])
    return {
        "frames": len(frames),
        "atoms_per_frame": sorted({len(atoms) for atoms in frames}),
        "chemical_formulas": sorted({atoms.get_chemical_formula() for atoms in frames}),
        "relative_energy_min_eV": float(energies.min()),
        "relative_energy_max_eV": float(energies.max()),
        "coordinate_min_A": float(coordinates.min()),
        "coordinate_max_A": float(coordinates.max()),
        "maximum_atomic_force_eV_per_A": float(maximum_forces.max()),
        "maximum_net_force_eV_per_A": float(net_forces.max()),
    }


def write_frames(path: Path, frames: Sequence[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, list(frames), format="extxyz")
    reloaded = read(path, index=":")
    if len(reloaded) != len(frames):
        raise ValueError(f"Round-trip frame count mismatch for {path}")
    for original, restored in zip(frames, reloaded):
        if restored.pbc.any():
            raise ValueError(f"Prepared molecular data became periodic in {path}")
        if abs(original.get_potential_energy() - restored.get_potential_energy()) > 1.0e-6:
            raise ValueError(f"Round-trip energy mismatch for {path}")
        if not np.allclose(original.get_forces(), restored.get_forces(), atol=1.0e-6, rtol=0.0):
            raise ValueError(f"Round-trip force mismatch for {path}")


def copy_structure(source: Path, destination: Path) -> None:
    frames = read(source, index=":")
    for atoms in frames:
        atoms.pbc = False
        atoms.set_cell(np.zeros((3, 3)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(destination, frames, format="extxyz")


def prepare_reaction(
    root: Path,
    cache: Path,
    reaction: str,
    files: Dict[str, SourceFile],
    coordinate: Callable[[np.ndarray, np.ndarray], Tuple[float, float, float]],
    coordinate_names: Sequence[str],
    primary_coordinate: str,
    charge: int,
    multiplicity: int,
    validation_fraction: float,
) -> dict:
    reaction_root = root / reaction
    data_root = reaction_root / "data"
    source_paths = {name: fetch(source, cache) for name, source in files.items()}

    training_archives = [source_paths["wtmetad"], source_paths["downhill"]]
    absolute_energies = []
    for archive_path in training_archives:
        with np.load(archive_path, allow_pickle=True) as archive:
            absolute_energies.extend(np.asarray(archive["E_true"], dtype=float).tolist())
    energy_reference = float(min(absolute_energies))

    wtmetad = load_npz_frames(
        source_paths["wtmetad"],
        "wtmetad_ib_active_learning",
        coordinate,
        coordinate_names,
        energy_reference,
        charge,
        multiplicity,
    )
    downhill = load_npz_frames(
        source_paths["downhill"],
        "downhill_active_learning",
        coordinate,
        coordinate_names,
        energy_reference,
        charge,
        multiplicity,
    )
    combined, duplicate_count = deduplicate([*wtmetad, *downhill])
    combined_train, combined_valid = stratified_validation_split(
        combined, primary_coordinate, validation_fraction
    )
    wtmetad_train, wtmetad_valid = stratified_validation_split(
        wtmetad, primary_coordinate, validation_fraction
    )
    downhill_train, downhill_valid = stratified_validation_split(
        downhill, primary_coordinate, validation_fraction
    )

    write_frames(data_root / "combined_all.xyz", combined)
    write_frames(data_root / "combined_train.xyz", combined_train)
    write_frames(data_root / "combined_valid.xyz", combined_valid)
    write_frames(data_root / "wtmetad_all.xyz", wtmetad)
    write_frames(data_root / "wtmetad_train.xyz", wtmetad_train)
    write_frames(data_root / "wtmetad_valid.xyz", wtmetad_valid)
    write_frames(data_root / "downhill_all.xyz", downhill)
    write_frames(data_root / "downhill_train.xyz", downhill_train)
    write_frames(data_root / "downhill_valid.xyz", downhill_valid)

    tests = {}
    for key in ("us_test", "irc_test"):
        if key not in source_paths:
            continue
        frames = load_npz_frames(
            source_paths[key],
            key,
            coordinate,
            coordinate_names,
            energy_reference,
            charge,
            multiplicity,
        )
        overlap = {geometry_fingerprint(atoms) for atoms in combined}.intersection(
            geometry_fingerprint(atoms) for atoms in frames
        )
        if overlap:
            raise ValueError(f"{reaction} {key} overlaps the training pool")
        write_frames(data_root / f"{key}.xyz", frames)
        tests[key] = frame_statistics(frames, primary_coordinate)

    structures_root = reaction_root / "structures"
    for key in ("reactant", "transition_state", "product", "irc_coordinates"):
        if key in source_paths:
            copy_structure(source_paths[key], structures_root / files[key].filename)

    output_files = sorted(data_root.glob("*.xyz")) + sorted(structures_root.glob("*.xyz"))
    return {
        "figshare_article": FIGSHARE_ARTICLE,
        "energy_label_convention": "E_true - energy_reference_eV",
        "energy_reference_eV": energy_reference,
        "charge": charge,
        "multiplicity": multiplicity,
        "coordinate_names": list(coordinate_names),
        "primary_coordinate": primary_coordinate,
        "source_files": {
            key: {
                "file_id": source.file_id,
                "filename": source.filename,
                "md5": source.md5,
            }
            for key, source in files.items()
        },
        "duplicates_removed_from_combined_pool": duplicate_count,
        "statistics": {
            "combined_all": frame_statistics(combined, primary_coordinate),
            "combined_train": frame_statistics(combined_train, primary_coordinate),
            "combined_valid": frame_statistics(combined_valid, primary_coordinate),
            "wtmetad_all": frame_statistics(wtmetad, primary_coordinate),
            "wtmetad_train": frame_statistics(wtmetad_train, primary_coordinate),
            "wtmetad_valid": frame_statistics(wtmetad_valid, primary_coordinate),
            "downhill_all": frame_statistics(downhill, primary_coordinate),
            "downhill_train": frame_statistics(downhill_train, primary_coordinate),
            "downhill_valid": frame_statistics(downhill_valid, primary_coordinate),
            **tests,
        },
        "sha256": {str(path.relative_to(reaction_root)): sha256sum(path) for path in output_files},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache = (args.cache or (root / ".download_cache")).resolve()
    manifests = {
        "r2_methyl_shift": prepare_reaction(
            root,
            cache,
            "r2_methyl_shift",
            R2_FILES,
            r2_coordinate,
            ("r1_C_C_A", "r2_C_C_A", "cv_r1_minus_r2_A"),
            "cv_r1_minus_r2_A",
            charge=0,
            multiplicity=1,
            validation_fraction=args.validation_fraction,
        ),
    }
    for reaction, manifest in manifests.items():
        path = root / reaction / "data_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Prepared {reaction}: {path}")

    if not args.keep_cache and args.cache is None:
        shutil.rmtree(cache)


if __name__ == "__main__":
    main()
