#!/usr/bin/env python3
"""Run TRACE Frenkel--Ladd free energies for delta and cubic CsPbI3."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "transformers-ace-matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import yaml
from ase.build import sort
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
from ase.neighborlist import neighbor_list

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ti_core import (  # noqa: E402
    EV_PER_FORMULA_TO_KJ_MOL,
    KB_EV_K,
    block_means,
    crossing_temperature,
    einstein_free_energy_per_atom,
    gibbs_helmholtz_curve,
    isobaric_enthalpy_eV,
    pressure_volume_energy_eV,
    read_named_csv,
    symmetric_switching_estimate,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "npt", "anchors", "msd", "ti", "analyze", "all"),
        help="Workflow stage to execute",
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--profile", choices=("pilot", "screening", "production"), default="pilot")
    parser.add_argument("--run-directory", type=Path, default=None)
    parser.add_argument("--phase", choices=("delta", "cubic"), default=None)
    parser.add_argument("--temperature", type=float, default=None, help="Run one configured NPT temperature")
    parser.add_argument("--replica", type=int, default=None, help="Run one configured TI replica")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent independent LAMMPS jobs")
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help="Override system.torch_threads for each LAMMPS process",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs for the selected stage")
    parser.add_argument("--dry-run", action="store_true", help="Write inputs without launching LAMMPS")
    return parser.parse_args()


def load_configuration(path: Path, profile_name: str) -> Tuple[dict, dict]:
    with path.expanduser().resolve().open() as handle:
        config = yaml.safe_load(handle)
    if profile_name not in config["profiles"]:
        raise KeyError("Unknown profile: {}".format(profile_name))
    profile = dict(config["profiles"][profile_name])
    temperatures = sorted(float(value) for value in profile["temperatures_K"])
    profile["temperatures_K"] = temperatures
    anchor = float(config["system"]["anchor_temperature_K"])
    if not any(math.isclose(anchor, value, abs_tol=1.0e-10) for value in temperatures):
        raise ValueError("The selected profile must include the anchor temperature")
    return config, profile


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        SCRIPT_DIR / "ti_core.py",
        REPO_ROOT / "transformers_ace" / "deploy.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def find_lammps(configured: str) -> Path:
    candidates = []
    if configured and configured != "auto":
        candidates.append(Path(configured).expanduser())
    if os.environ.get("LAMMPS_COMMAND"):
        candidates.append(Path(os.environ["LAMMPS_COMMAND"]).expanduser())
    for name in ("lmp", "lmp_mpi", "lammps"):
        executable = shutil.which(name)
        if executable:
            candidates.append(Path(executable))
    candidates.append(Path.home() / "Documents/lammps_plumed/lammps/build/lmp")
    for candidate in candidates:
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "No LAMMPS executable was found. Set LAMMPS_COMMAND or system.lammps_executable in config.yaml."
    )


def phase_names(args: argparse.Namespace, config: dict) -> List[str]:
    return [args.phase] if args.phase else list(config["phases"].keys())


def temperatures(args: argparse.Namespace, profile: dict) -> List[float]:
    values = list(profile["temperatures_K"])
    if args.temperature is None:
        return values
    matches = [value for value in values if math.isclose(value, args.temperature, abs_tol=1.0e-10)]
    if not matches:
        raise ValueError("Requested temperature is not in the selected profile")
    return matches


def temperature_tag(temperature_K: float) -> str:
    if float(temperature_K).is_integer():
        return "{}K".format(int(temperature_K))
    return "{:g}K".format(temperature_K)


def threads_per_job(args: argparse.Namespace, config: dict) -> int:
    value = args.threads_per_job
    return int(config["system"]["torch_threads"] if value is None else value)


def lammps_preamble(data_path: Path, model_path: Path, timestep_ps: float) -> str:
    return """units metal
atom_style atomic
boundary p p p
read_data {data}

pair_style transformers_ace
pair_coeff * * {model} Cs Pb I
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
timestep {timestep:.12g}

group Cs type 1
group Pb type 2
group I type 3
compute temperature_com all temp/com
""".format(data=data_path.resolve(), model=model_path.resolve(), timestep=timestep_ps)


def write_lammps_data(path: Path, atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(
        str(path),
        atoms,
        format="lammps-data",
        specorder=["Cs", "Pb", "I"],
        masses=True,
        atom_style="atomic",
        units="metal",
        force_skew=True,
    )


def prepare(args: argparse.Namespace, config: dict, profile: dict, run_root: Path) -> dict:
    checkpoint = resolve_repo_path(config["system"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    expected = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "configuration_sha256": mapping_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "profile": args.profile,
        "profile_settings": profile,
        "pressure_bar": float(config["system"]["pressure_bar"]),
        "anchor_temperature_K": float(config["system"]["anchor_temperature_K"]),
        "type_map": list(config["system"]["type_map"]),
    }
    manifest_path = run_root / "manifest.json"

    if manifest_path.exists() and not args.overwrite:
        current = json.loads(manifest_path.read_text())
        for key in (
            "checkpoint_sha256",
            "configuration_sha256",
            "implementation_sha256",
            "profile",
            "profile_settings",
        ):
            if current.get(key) != expected.get(key):
                raise RuntimeError(
                    "Existing run metadata differs from the requested calculation. "
                    "Use a new --run-directory or rerun prepare with --overwrite."
                )
        return current

    if args.overwrite and run_root.exists():
        shutil.rmtree(str(run_root))
    run_root.mkdir(parents=True, exist_ok=True)

    model_path = run_root / "model.transformers_ace.pt"
    from transformers_ace.deploy import export_lammps_model

    example = resolve_repo_path(config["phases"]["cubic"]["structure"])
    export_lammps_model(
        checkpoint=checkpoint,
        output=model_path,
        type_map=list(config["system"]["type_map"]),
        example_structure=example,
        device="cpu",
    )

    phase_records = {}
    expected_natoms = int(config["system"].get("expected_natoms", 240))
    if expected_natoms <= 0 or expected_natoms % 5 != 0:
        raise ValueError("system.expected_natoms must be a positive multiple of 5")
    expected_formula_units = expected_natoms // 5
    expected_counts = {
        "Cs": expected_formula_units,
        "Pb": expected_formula_units,
        "I": 3 * expected_formula_units,
    }
    for phase, phase_config in config["phases"].items():
        source = resolve_repo_path(phase_config["structure"])
        atoms = read(str(source))
        atoms.pbc = True
        atoms = sort(atoms.repeat(tuple(int(value) for value in phase_config["repeat"])))
        if len(atoms) != expected_natoms:
            raise ValueError(
                "{} repeat produces {} atoms, not {}".format(
                    phase, len(atoms), expected_natoms
                )
            )
        counts = {symbol: atoms.get_chemical_symbols().count(symbol) for symbol in ("Cs", "Pb", "I")}
        if counts != expected_counts:
            raise ValueError("Unexpected {} composition: {}".format(phase, counts))
        if min(atoms.cell.lengths()) <= 12.0:
            raise ValueError("{} cell is too short for a 6 Angstrom cutoff".format(phase))

        structure_path = run_root / "structures" / "{}.extxyz".format(phase)
        data_path = run_root / "structures" / "{}.data".format(phase)
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(structure_path), atoms, format="extxyz")
        write_lammps_data(data_path, atoms)
        phase_records[phase] = {
            "label": phase_config["label"],
            "source": str(source),
            "repeat": list(phase_config["repeat"]),
            "barostat": phase_config["barostat"],
            "natoms": len(atoms),
            "counts": counts,
            "cell_A": atoms.cell.array.tolist(),
            "volume_A3": float(atoms.get_volume()),
            "structure": str(structure_path.resolve()),
            "lammps_data": str(data_path.resolve()),
        }

    expected["model"] = str(model_path.resolve())
    expected["model_sha256"] = sha256(model_path)
    expected["phases"] = phase_records
    json_write(manifest_path, expected)
    return expected


def require_manifest(run_root: Path, config: dict, profile_name: str, profile: dict) -> dict:
    path = run_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run the prepare stage first: {}".format(path))
    manifest = json.loads(path.read_text())
    checkpoint = resolve_repo_path(config["system"]["checkpoint"])
    current = {
        "checkpoint_sha256": sha256(checkpoint),
        "configuration_sha256": mapping_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "profile": profile_name,
        "profile_settings": profile,
    }
    for key, value in current.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                "The run manifest is stale for {}. Use a new --run-directory or rerun "
                "the prepare stage with --overwrite.".format(key)
            )
    return manifest


def npt_input(
    data_path: Path,
    model_path: Path,
    temperature_K: float,
    pressure_bar: float,
    barostat: str,
    config: dict,
    profile: dict,
    seed: int,
) -> str:
    sample_every = int(profile["npt_sample_every"])
    preamble = lammps_preamble(data_path, model_path, float(config["system"]["timestep_ps"]))
    return preamble + """
velocity all create {temperature:.12g} {seed} mom yes rot no dist gaussian
fix ensemble all npt temp {temperature:.12g} {temperature:.12g} {tdamp:.12g} {barostat} {pressure:.12g} {pressure:.12g} {pdamp:.12g} tchain 3 pchain 3 mtk yes
fix_modify ensemble temp temperature_com

thermo_style custom step c_temperature_com pe ke etotal enthalpy press vol lx ly lz xy xz yz
thermo_modify temp temperature_com flush yes
thermo {thermo_every}
run {equil_steps}

reset_timestep 0
variable sample_step equal step
variable sample_temp equal c_temperature_com
variable sample_pe equal pe
variable sample_ke equal ke
variable sample_h equal enthalpy
variable sample_press equal press
variable sample_vol equal vol
variable sample_lx equal lx
variable sample_ly equal ly
variable sample_lz equal lz
variable sample_xy equal xy
variable sample_xz equal xz
variable sample_yz equal yz
fix samples all print {sample_every} "${{sample_step}},${{sample_temp}},${{sample_pe}},${{sample_ke}},${{sample_h}},${{sample_press}},${{sample_vol}},${{sample_lx}},${{sample_ly}},${{sample_lz}},${{sample_xy}},${{sample_xz}},${{sample_yz}}" file thermo.csv screen no title "step,temp_K,pe_eV,ke_eV,lammps_enthalpy_eV,pressure_bar,volume_A3,lx_A,ly_A,lz_A,xy_A,xz_A,yz_A"
run {sample_steps}
unfix samples
write_data final.data
""".format(
        temperature=temperature_K,
        seed=seed,
        tdamp=float(config["thermostat"]["npt_temperature_damping_ps"]),
        barostat=barostat,
        pressure=pressure_bar,
        pdamp=float(config["thermostat"]["npt_pressure_damping_ps"]),
        thermo_every=max(1, sample_every * 10),
        equil_steps=int(profile["npt_equil_steps"]),
        sample_every=sample_every,
        sample_steps=int(profile["npt_sample_steps"]),
    )


def msd_input(
    data_path: Path,
    model_path: Path,
    temperature_K: float,
    config: dict,
    profile: dict,
    seed: int,
) -> str:
    sample_every = int(profile["msd_sample_every"])
    preamble = lammps_preamble(data_path, model_path, float(config["system"]["timestep_ps"]))
    return preamble + """
compute msd_Cs Cs msd com no
compute msd_Pb Pb msd com no
compute msd_I I msd com no

velocity all create {temperature:.12g} {seed} mom yes rot no dist gaussian
fix integrate all nve
fix bath all langevin {temperature:.12g} {temperature:.12g} {damping:.12g} {bath_seed} zero yes
fix_modify bath temp temperature_com

thermo_style custom step c_temperature_com pe etotal c_msd_Cs[4] c_msd_Pb[4] c_msd_I[4]
thermo {thermo_every}
run {equil_steps}

variable sample_step equal step
variable sample_temp equal c_temperature_com
variable sample_Cs equal c_msd_Cs[4]
variable sample_Pb equal c_msd_Pb[4]
variable sample_I equal c_msd_I[4]
fix samples all print {sample_every} "${{sample_step}},${{sample_temp}},${{sample_Cs}},${{sample_Pb}},${{sample_I}}" file msd.csv screen no title "step,temp_K,msd_Cs_A2,msd_Pb_A2,msd_I_A2"
run {sample_steps}
unfix samples
write_data final.data
""".format(
        temperature=temperature_K,
        seed=seed,
        damping=float(config["thermostat"]["langevin_damping_ps"]),
        bath_seed=seed + 104729,
        thermo_every=max(1, sample_every * 10),
        equil_steps=int(profile["msd_equil_steps"]),
        sample_every=sample_every,
        sample_steps=int(profile["msd_sample_steps"]),
    )


def ti_input(
    data_path: Path,
    model_path: Path,
    temperature_K: float,
    springs: Mapping[str, float],
    config: dict,
    profile: dict,
    seed: int,
) -> str:
    equil_steps = int(profile["ti_equil_steps"])
    switch_steps = int(profile["ti_switch_steps"])
    preamble = lammps_preamble(data_path, model_path, float(config["system"]["timestep_ps"]))
    return preamble + """
velocity all create {temperature:.12g} {seed} mom yes rot no dist gaussian

# Ordering is essential: interpolate the physical forces before thermostatting.
fix integrate all nve
fix ti_Cs Cs ti/spring {spring_Cs:.16g} {switch_steps} {equil_steps} function 2
fix ti_Pb Pb ti/spring {spring_Pb:.16g} {switch_steps} {equil_steps} function 2
fix ti_I I ti/spring {spring_I:.16g} {switch_steps} {equil_steps} function 2
fix bath all langevin {temperature:.12g} {temperature:.12g} {damping:.12g} {bath_seed} zero yes
fix_modify bath temp temperature_com

variable sample_step equal step
variable spring_energy equal f_ti_Cs+f_ti_Pb+f_ti_I
variable dU equal pe-v_spring_energy
variable coupling equal f_ti_Cs[1]
variable coupling_rate equal f_ti_Cs[2]
thermo_style custom step c_temperature_com pe v_spring_energy v_dU v_coupling
thermo {thermo_every}

run {equil_steps}
fix work all print 1 "${{sample_step}} ${{dU}} ${{coupling}} ${{coupling_rate}}" screen no file forward.dat title "# step U_real_minus_U_Einstein_eV lambda dlambda_per_step"
run {switch_steps}
unfix work

run {equil_steps}
fix work all print 1 "${{sample_step}} ${{dU}} ${{coupling}} ${{coupling_rate}}" screen no file backward.dat title "# step U_real_minus_U_Einstein_eV lambda dlambda_per_step"
run {switch_steps}
unfix work
write_data final.data
""".format(
        temperature=temperature_K,
        seed=seed,
        spring_Cs=float(springs["Cs"]),
        spring_Pb=float(springs["Pb"]),
        spring_I=float(springs["I"]),
        switch_steps=switch_steps,
        equil_steps=equil_steps,
        damping=float(config["thermostat"]["langevin_damping_ps"]),
        bath_seed=seed + 104729,
        thermo_every=max(10, switch_steps // 20),
    )


def output_complete(directory: Path, expected_files: Sequence[str]) -> bool:
    status = directory / "run_status.json"
    if not status.exists():
        return False
    try:
        record = json.loads(status.read_text())
    except (ValueError, OSError):
        return False
    return record.get("state") == "complete" and all((directory / name).exists() for name in expected_files)


def execute_lammps(
    directory: Path,
    input_text: str,
    lammps: Path,
    expected_files: Sequence[str],
    threads: int,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    if output_complete(directory, expected_files) and not overwrite:
        return {"directory": str(directory), "state": "skipped"}
    if overwrite and directory.exists():
        shutil.rmtree(str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "in.lammps"
    input_path.write_text("log log.lammps\n" + input_text)
    if dry_run:
        return {"directory": str(directory), "state": "dry-run"}

    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[variable] = str(max(1, threads))
    started = time.time()
    with (directory / "stdout.log").open("w") as stdout:
        result = subprocess.run(
            [str(lammps), "-in", input_path.name],
            cwd=str(directory),
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
        )
    state = "complete" if result.returncode == 0 and all((directory / name).exists() for name in expected_files) else "failed"
    record = {
        "directory": str(directory),
        "state": state,
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - started,
        "expected_files": list(expected_files),
    }
    json_write(directory / "run_status.json", record)
    if state != "complete":
        raise RuntimeError("LAMMPS failed in {}; inspect stdout.log".format(directory))
    return record


def run_tasks(tasks: Sequence[Tuple], workers: int) -> None:
    if workers <= 1:
        for task in tasks:
            print(execute_lammps(*task))
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute_lammps, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())


def run_npt(args: argparse.Namespace, config: dict, profile: dict, run_root: Path) -> None:
    manifest = require_manifest(run_root, config, args.profile, profile)
    lammps = find_lammps(str(config["system"].get("lammps_executable", "auto")))
    tasks = []
    base_seed = int(config["system"]["seed"])
    for phase_index, phase in enumerate(phase_names(args, config)):
        phase_record = manifest["phases"][phase]
        for temperature_index, temperature_K in enumerate(temperatures(args, profile)):
            directory = run_root / "npt" / phase / temperature_tag(temperature_K)
            text = npt_input(
                data_path=Path(phase_record["lammps_data"]),
                model_path=Path(manifest["model"]),
                temperature_K=temperature_K,
                pressure_bar=float(config["system"]["pressure_bar"]),
                barostat=phase_record["barostat"],
                config=config,
                profile=profile,
                seed=base_seed + 1000 * phase_index + 17 * temperature_index,
            )
            tasks.append(
                (
                    directory,
                    text,
                    lammps,
                    ("thermo.csv", "final.data"),
                    threads_per_job(args, config),
                    args.overwrite,
                    args.dry_run,
                )
            )
    run_tasks(tasks, args.workers)


def build_anchors(args: argparse.Namespace, config: dict, profile: dict, run_root: Path) -> None:
    manifest = require_manifest(run_root, config, args.profile, profile)
    anchor_temperature = float(config["system"]["anchor_temperature_K"])
    for phase in phase_names(args, config):
        thermo_path = run_root / "npt" / phase / temperature_tag(anchor_temperature) / "thermo.csv"
        table = read_named_csv(thermo_path)
        mean_cell = np.diag(
            [float(np.mean(table[name])) for name in ("lx_A", "ly_A", "lz_A")]
        )
        if any(abs(float(np.mean(table[name]))) > 1.0e-6 for name in ("xy_A", "xz_A", "yz_A")):
            raise ValueError("The current anchor builder expects orthogonal cells")

        mean_volume = float(np.mean(table["volume_A3"]))
        mean_cell *= (mean_volume / float(np.linalg.det(mean_cell))) ** (1.0 / 3.0)
        atoms = read(manifest["phases"][phase]["structure"])
        atoms.set_cell(mean_cell, scale_atoms=True)
        atoms.wrap()
        anchor_directory = run_root / "anchor" / phase
        anchor_directory.mkdir(parents=True, exist_ok=True)
        write(str(anchor_directory / "anchor.extxyz"), atoms, format="extxyz")
        write_lammps_data(anchor_directory / "anchor.data", atoms)
        symbols = atoms.get_chemical_symbols()
        record = {
            "phase": phase,
            "temperature_K": anchor_temperature,
            "natoms": len(atoms),
            "cell_A": atoms.cell.array.tolist(),
            "volume_A3": float(atoms.get_volume()),
            "mean_npt_volume_A3": mean_volume,
            "std_npt_volume_A3": float(np.std(table["volume_A3"], ddof=1)) if len(table["volume_A3"]) > 1 else 0.0,
            "masses_amu": [float(atomic_masses[atomic_numbers[symbol]]) for symbol in symbols],
            "symbols": symbols,
        }
        json_write(anchor_directory / "anchor.json", record)


def run_msd(args: argparse.Namespace, config: dict, profile: dict, run_root: Path) -> None:
    manifest = require_manifest(run_root, config, args.profile, profile)
    lammps = find_lammps(str(config["system"].get("lammps_executable", "auto")))
    anchor_temperature = float(config["system"]["anchor_temperature_K"])
    tasks = []
    base_seed = int(config["system"]["seed"]) + 30000
    for phase_index, phase in enumerate(phase_names(args, config)):
        directory = run_root / "anchor" / phase / "msd"
        text = msd_input(
            data_path=run_root / "anchor" / phase / "anchor.data",
            model_path=Path(manifest["model"]),
            temperature_K=anchor_temperature,
            config=config,
            profile=profile,
            seed=base_seed + 101 * phase_index,
        )
        tasks.append(
            (
                directory,
                text,
                lammps,
                ("msd.csv", "final.data"),
                threads_per_job(args, config),
                args.overwrite,
                args.dry_run,
            )
        )
    run_tasks(tasks, args.workers)
    if args.dry_run:
        return

    minimum = float(config["einstein_reference"]["minimum_spring_eV_A2"])
    maximum = float(config["einstein_reference"]["maximum_spring_eV_A2"])
    default = float(config["einstein_reference"]["default_spring_eV_A2"])
    for phase in phase_names(args, config):
        table = read_named_csv(run_root / "anchor" / phase / "msd" / "msd.csv")
        springs = {}
        mean_msd = {}
        for symbol in ("Cs", "Pb", "I"):
            value = float(np.mean(table["msd_{}_A2".format(symbol)]))
            mean_msd[symbol] = value
            estimate = 3.0 * KB_EV_K * anchor_temperature / value if value > 0.0 else default
            springs[symbol] = float(np.clip(estimate, minimum, maximum))
        json_write(
            run_root / "anchor" / phase / "springs.json",
            {
                "phase": phase,
                "temperature_K": anchor_temperature,
                "mean_msd_A2": mean_msd,
                "spring_eV_A2": springs,
                "estimator": "k_s = 3 k_B T / <|r-r_0|^2>_s",
                "clip_eV_A2": [minimum, maximum],
            },
        )


def run_ti(args: argparse.Namespace, config: dict, profile: dict, run_root: Path) -> None:
    manifest = require_manifest(run_root, config, args.profile, profile)
    lammps = find_lammps(str(config["system"].get("lammps_executable", "auto")))
    anchor_temperature = float(config["system"]["anchor_temperature_K"])
    replica_indices = [args.replica] if args.replica is not None else list(range(int(profile["ti_replicas"])))
    for replica in replica_indices:
        if replica < 0 or replica >= int(profile["ti_replicas"]):
            raise ValueError("Replica index is outside the selected profile")
    tasks = []
    base_seed = int(config["system"]["seed"]) + 60000
    for phase_index, phase in enumerate(phase_names(args, config)):
        springs_path = run_root / "anchor" / phase / "springs.json"
        if not springs_path.exists():
            raise FileNotFoundError("Run the msd stage first: {}".format(springs_path))
        springs = json.loads(springs_path.read_text())["spring_eV_A2"]
        for replica in replica_indices:
            directory = run_root / "ti" / phase / "replica_{:02d}".format(replica)
            text = ti_input(
                data_path=run_root / "anchor" / phase / "anchor.data",
                model_path=Path(manifest["model"]),
                temperature_K=anchor_temperature,
                springs=springs,
                config=config,
                profile=profile,
                seed=base_seed + 1009 * phase_index + 37 * replica,
            )
            tasks.append(
                (
                    directory,
                    text,
                    lammps,
                    ("forward.dat", "backward.dat", "final.data"),
                    threads_per_job(args, config),
                    args.overwrite,
                    args.dry_run,
                )
            )
    run_tasks(tasks, args.workers)


def resampled_mean(values: np.ndarray, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0])
    return float(np.mean(rng.choice(values, size=len(values), replace=True)))


def pair_fingerprint(atoms, cutoff_A: float, bin_width_A: float) -> np.ndarray:
    symbols = np.asarray(atoms.get_chemical_symbols())
    i_atom, j_atom, distance = neighbor_list("ijd", atoms, cutoff_A)
    unique = i_atom < j_atom
    i_atom, j_atom, distance = i_atom[unique], j_atom[unique], distance[unique]
    edges = np.arange(0.0, cutoff_A + bin_width_A, bin_width_A)
    channels = []
    for first, second in (("Cs", "Cs"), ("Cs", "Pb"), ("Cs", "I"), ("Pb", "Pb"), ("Pb", "I"), ("I", "I")):
        selected = ((symbols[i_atom] == first) & (symbols[j_atom] == second)) | (
            (symbols[i_atom] == second) & (symbols[j_atom] == first)
        )
        histogram = np.histogram(distance[selected], bins=edges)[0].astype(float)
        # Smooth both thermal and zero-temperature fingerprints on a 0.15 A scale.
        sigma_bins = max(1.0, 0.15 / bin_width_A)
        radius = int(math.ceil(4.0 * sigma_bins))
        grid = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        histogram = np.convolve(histogram, kernel, mode="same")
        norm = np.linalg.norm(histogram)
        channels.append(histogram / norm if norm > 0.0 else histogram)
    return np.concatenate(channels)


def cosine_distance(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(1.0 - np.dot(first, second) / denominator) if denominator > 0.0 else float("nan")


def structural_diagnostics(config: dict, profile: dict, run_root: Path, manifest: dict) -> List[dict]:
    cutoff = float(config["analysis"]["rdf_cutoff_A"])
    bin_width = float(config["analysis"]["rdf_bin_width_A"])
    coordination_cutoff = float(config["analysis"]["pb_i_coordination_cutoff_A"])
    references = {phase: read(record["structure"]) for phase, record in manifest["phases"].items()}
    output = []
    for phase in manifest["phases"]:
        for temperature_K in profile["temperatures_K"]:
            final_path = run_root / "npt" / phase / temperature_tag(temperature_K) / "final.data"
            if not final_path.exists():
                continue
            atoms = read(
                str(final_path),
                format="lammps-data",
                atom_style="atomic",
                Z_of_type={1: atomic_numbers["Cs"], 2: atomic_numbers["Pb"], 3: atomic_numbers["I"]},
            )
            atoms.pbc = True
            final_fingerprint = pair_fingerprint(atoms, cutoff, bin_width)
            distances = {}
            for reference_phase, reference in references.items():
                scaled = reference.copy()
                scaled.set_cell(atoms.cell, scale_atoms=True)
                distances[reference_phase] = cosine_distance(
                    final_fingerprint, pair_fingerprint(scaled, cutoff, bin_width)
                )
            i_atom, j_atom = neighbor_list("ij", atoms, coordination_cutoff)
            symbols = np.asarray(atoms.get_chemical_symbols())
            selected = (symbols[i_atom] == "Pb") & (symbols[j_atom] == "I")
            coordination = np.bincount(i_atom[selected], minlength=len(atoms))[symbols == "Pb"]
            output.append(
                {
                    "phase": phase,
                    "temperature_K": temperature_K,
                    "mean_Pb_I_coordination": float(np.mean(coordination)),
                    "fingerprint_distance_to_delta": distances["delta"],
                    "fingerprint_distance_to_cubic": distances["cubic"],
                    "closest_reference": min(distances, key=distances.get),
                }
            )
    return output


def analyze(args: argparse.Namespace, config: dict, profile: dict, run_root: Path) -> None:
    import matplotlib.pyplot as plt

    manifest = require_manifest(run_root, config, args.profile, profile)
    phases = ("delta", "cubic")
    temperature_grid = np.asarray(profile["temperatures_K"], dtype=float)
    anchor_temperature = float(config["system"]["anchor_temperature_K"])
    pressure_bar = float(config["system"]["pressure_bar"])
    timestep_ps = float(config["system"]["timestep_ps"])
    formula_atoms = int(config["analysis"]["formula_atoms"])
    conversion = formula_atoms * EV_PER_FORMULA_TO_KJ_MOL

    sample_interval_ps = timestep_ps * int(profile["npt_sample_every"])
    samples_per_block = max(1, int(round(float(profile["block_length_ps"]) / sample_interval_ps)))
    npt_blocks = {}
    npt_rows = []
    for phase in phases:
        natoms = int(manifest["phases"][phase]["natoms"])
        for temperature_K in temperature_grid:
            table = read_named_csv(run_root / "npt" / phase / temperature_tag(temperature_K) / "thermo.csv")
            physical_enthalpy = isobaric_enthalpy_eV(
                table["pe_eV"],
                table["ke_eV"],
                table["volume_A3"],
                pressure_bar,
            )
            h_blocks = block_means(physical_enthalpy / natoms, samples_per_block)
            v_blocks = block_means(table["volume_A3"], samples_per_block)
            p_blocks = block_means(table["pressure_bar"], samples_per_block)
            npt_blocks[(phase, float(temperature_K))] = {"enthalpy": h_blocks, "volume": v_blocks}
            h_error = float(np.std(h_blocks, ddof=1) / math.sqrt(len(h_blocks))) if len(h_blocks) > 1 else float("nan")
            p_error = float(np.std(p_blocks, ddof=1) / math.sqrt(len(p_blocks))) if len(p_blocks) > 1 else float("nan")
            npt_rows.append(
                {
                    "phase": phase,
                    "temperature_K": temperature_K,
                    "samples": len(table["pe_eV"]),
                    "blocks": len(h_blocks),
                    "enthalpy_eV_atom": float(np.mean(h_blocks)),
                    "enthalpy_sem_eV_atom": h_error,
                    "volume_A3": float(np.mean(v_blocks)),
                    "pressure_bar": float(np.mean(p_blocks)),
                    "pressure_sem_bar": p_error,
                    "temperature_mean_K": float(np.mean(table["temp_K"])),
                }
            )

    phase_data = {}
    anchor_rows = []
    for phase in phases:
        anchor = json.loads((run_root / "anchor" / phase / "anchor.json").read_text())
        spring_record = json.loads((run_root / "anchor" / phase / "springs.json").read_text())
        springs_by_symbol = spring_record["spring_eV_A2"]
        springs_per_atom = [float(springs_by_symbol[symbol]) for symbol in anchor["symbols"]]
        masses = np.asarray(anchor["masses_amu"], dtype=float)
        work_records = []
        for replica in range(int(profile["ti_replicas"])):
            replica_directory = run_root / "ti" / phase / "replica_{:02d}".format(replica)
            work = symmetric_switching_estimate(
                replica_directory / "forward.dat", replica_directory / "backward.dat"
            )
            work["replica"] = replica
            work_records.append(work)
        work_values = np.asarray([record["reversible_eV"] for record in work_records])
        anchor_volumes = npt_blocks[(phase, anchor_temperature)]["volume"]
        anchor_volume = float(anchor["volume_A3"])
        f_einstein = einstein_free_energy_per_atom(
            anchor_temperature, anchor_volume, masses, springs_per_atom
        )
        natoms = len(masses)
        g_anchor = (
            f_einstein
            + float(np.mean(work_values)) / natoms
            + pressure_volume_energy_eV(pressure_bar, anchor_volume) / natoms
        )
        enthalpy = np.asarray(
            [np.mean(npt_blocks[(phase, float(value))]["enthalpy"]) for value in temperature_grid]
        )
        gibbs = gibbs_helmholtz_curve(temperature_grid, enthalpy, anchor_temperature, g_anchor)
        phase_data[phase] = {
            "anchor": anchor,
            "masses": masses,
            "springs": springs_per_atom,
            "work": work_records,
            "work_values": work_values,
            "anchor_volumes": anchor_volumes,
            "f_einstein": f_einstein,
            "g_anchor": g_anchor,
            "enthalpy": enthalpy,
            "gibbs": gibbs,
        }
        for record in work_records:
            anchor_rows.append(
                {
                    "phase": phase,
                    "replica": record["replica"],
                    "forward_eV": record["forward_eV"],
                    "backward_oriented_eV": record["backward_oriented_eV"],
                    "reversible_eV": record["reversible_eV"],
                    "hysteresis_eV": record["hysteresis_eV"],
                    "hysteresis_meV_atom": 1000.0 * record["hysteresis_eV"] / natoms,
                    "spring_Cs_eV_A2": springs_by_symbol["Cs"],
                    "spring_Pb_eV_A2": springs_by_symbol["Pb"],
                    "spring_I_eV_A2": springs_by_symbol["I"],
                    "f_Einstein_eV_atom": f_einstein,
                    "g_anchor_eV_atom": g_anchor,
                }
            )

    n_bootstrap = int(profile["bootstrap_samples"])
    rng = np.random.default_rng(int(config["system"]["seed"]) + 90000)
    bootstrap_g = {phase: np.empty((n_bootstrap, len(temperature_grid))) for phase in phases}
    for sample in range(n_bootstrap):
        for phase in phases:
            data = phase_data[phase]
            volume = float(data["anchor"]["volume_A3"])
            f_einstein = float(data["f_einstein"])
            work = resampled_mean(data["work_values"], rng)
            natoms = len(data["masses"])
            g_anchor = (
                f_einstein
                + work / natoms
                + pressure_volume_energy_eV(pressure_bar, volume) / natoms
            )
            enthalpy = np.asarray(
                [
                    resampled_mean(npt_blocks[(phase, float(value))]["enthalpy"], rng)
                    for value in temperature_grid
                ]
            )
            bootstrap_g[phase][sample] = gibbs_helmholtz_curve(
                temperature_grid, enthalpy, anchor_temperature, g_anchor
            )

    confidence = float(config["analysis"]["confidence_level"])
    alpha = 0.5 * (1.0 - confidence)
    delta_g = phase_data["delta"]["gibbs"] - phase_data["cubic"]["gibbs"]
    bootstrap_delta = bootstrap_g["delta"] - bootstrap_g["cubic"]
    delta_low, delta_high = np.quantile(bootstrap_delta, [alpha, 1.0 - alpha], axis=0)
    crossing = crossing_temperature(temperature_grid, delta_g)
    bootstrap_crossings = np.asarray(
        [crossing_temperature(temperature_grid, row) for row in bootstrap_delta], dtype=float
    )
    finite_crossings = bootstrap_crossings[np.isfinite(bootstrap_crossings)]

    results_directory = run_root / "results"
    results_directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        results_directory / "npt_summary.csv",
        list(npt_rows[0].keys()),
        npt_rows,
    )
    write_csv(
        results_directory / "anchor_switching.csv",
        list(anchor_rows[0].keys()),
        anchor_rows,
    )
    phase_rows = []
    for index, temperature_K in enumerate(temperature_grid):
        phase_rows.append(
            {
                "temperature_K": temperature_K,
                "g_delta_eV_atom": phase_data["delta"]["gibbs"][index],
                "g_cubic_eV_atom": phase_data["cubic"]["gibbs"][index],
                "delta_g_delta_minus_cubic_eV_atom": delta_g[index],
                "delta_g_delta_minus_cubic_kJ_mol_fu": delta_g[index] * conversion,
                "ci_low_kJ_mol_fu": delta_low[index] * conversion,
                "ci_high_kJ_mol_fu": delta_high[index] * conversion,
            }
        )
    write_csv(results_directory / "phase_diagram.csv", list(phase_rows[0].keys()), phase_rows)

    diagnostics = structural_diagnostics(config, profile, run_root, manifest)
    if diagnostics:
        write_csv(
            results_directory / "structural_diagnostics.csv",
            list(diagnostics[0].keys()),
            diagnostics,
        )

    quality_issues = []
    if manifest["profile"] == "pilot":
        quality_issues.append("pilot trajectories are intentionally too short for thermodynamics")
    minimum_blocks = int(config["analysis"]["minimum_independent_blocks"])
    temperature_tolerance = float(config["analysis"]["temperature_relative_tolerance"])
    pressure_tolerance = float(config["analysis"]["pressure_mean_tolerance_bar"])
    hysteresis_tolerance = float(config["analysis"]["maximum_switching_hysteresis_meV_atom"])
    for row in npt_rows:
        key = "{} at {:g} K".format(row["phase"], row["temperature_K"])
        if int(row["blocks"]) < minimum_blocks:
            quality_issues.append("{} has only {} independent blocks".format(key, row["blocks"]))
        if abs(float(row["temperature_mean_K"]) - float(row["temperature_K"])) > (
            temperature_tolerance * float(row["temperature_K"])
        ):
            quality_issues.append("{} has an unconverged mean temperature".format(key))
        if abs(float(row["pressure_bar"]) - pressure_bar) > pressure_tolerance:
            quality_issues.append("{} has an unconverged mean pressure".format(key))
    for row in anchor_rows:
        if abs(float(row["hysteresis_meV_atom"])) > hysteresis_tolerance:
            quality_issues.append(
                "{} replica {} exceeds the switching-hysteresis tolerance".format(
                    row["phase"], row["replica"]
                )
            )
    for row in diagnostics:
        if row["closest_reference"] != row["phase"]:
            quality_issues.append(
                "{} at {:g} K is closer to the {} structural fingerprint".format(
                    row["phase"], row["temperature_K"], row["closest_reference"]
                )
            )
        if not 5.5 <= float(row["mean_Pb_I_coordination"]) <= 6.5:
            quality_issues.append(
                "{} at {:g} K has non-octahedral mean Pb-I coordination".format(
                    row["phase"], row["temperature_K"]
                )
            )
    scientifically_interpretable = len(quality_issues) == 0

    crossing_interval = [None, None]
    if len(finite_crossings):
        crossing_interval = np.quantile(finite_crossings, [alpha, 1.0 - alpha]).tolist()
    report = {
        "profile": manifest["profile"],
        "pressure_bar": pressure_bar,
        "anchor_temperature_K": anchor_temperature,
        "transition_temperature_K": float(crossing) if np.isfinite(crossing) else None,
        "transition_temperature_confidence_interval_K": crossing_interval,
        "bootstrap_samples_with_crossing": int(len(finite_crossings)),
        "bootstrap_samples": n_bootstrap,
        "delta_g_definition": "G_delta - G_cubic",
        "enthalpy_definition": "mean(E_potential + E_kinetic + P_external V)",
        "finite_size_atoms": int(manifest["phases"]["delta"]["natoms"]),
        "classical_nuclei": True,
        "scientifically_interpretable": scientifically_interpretable,
        "quality_issues": quality_issues,
    }
    json_write(results_directory / "report.json", report)

    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.labelsize": 32,
            "axes.titlesize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.linewidth": 1.2,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), constrained_layout=True)
    colors = {"delta": "#2f6f9f", "cubic": "#d46a3a"}
    labels = {"delta": r"$\delta$-CsPbI$_3$", "cubic": r"$\alpha$-CsPbI$_3$"}
    reference_bootstrap = bootstrap_g["cubic"][:, [0]]
    reference_value = phase_data["cubic"]["gibbs"][0]
    for phase in phases:
        relative = (phase_data[phase]["gibbs"] - reference_value) * conversion
        relative_bootstrap = (bootstrap_g[phase] - reference_bootstrap) * conversion
        low, high = np.quantile(relative_bootstrap, [alpha, 1.0 - alpha], axis=0)
        axes[0].plot(temperature_grid, relative, "o-", lw=2.0, ms=5.5, color=colors[phase], label=labels[phase])
        axes[0].fill_between(temperature_grid, low, high, color=colors[phase], alpha=0.18, linewidth=0)
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel(r"$G(T)-G_{\alpha}(300\,\mathrm{K})$ (kJ mol$^{-1}$ f.u.$^{-1}$)")
    axes[0].legend(frameon=False)

    delta_kj = delta_g * conversion
    experimental_transition_K = 600.0
    axes[1].axhline(0.0, color="0.25", lw=1.1)
    axes[1].plot(temperature_grid, delta_kj, "o-", color="#6a3d9a", lw=2.2, ms=5.5)
    axes[1].fill_between(
        temperature_grid,
        delta_low * conversion,
        delta_high * conversion,
        color="#6a3d9a",
        alpha=0.18,
        linewidth=0,
    )
    if np.isfinite(crossing):
        axes[1].axvline(crossing, color="0.35", ls="--", lw=1.2)
        axes[1].annotate(
            r"$T_{\mathrm{TRACE}}=%.0f$ K" % crossing,
            xy=(crossing, 0.0),
            xytext=(crossing - 18.0, 1.15),
            textcoords="data",
            ha="right",
            va="bottom",
            fontsize=22,
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 1.0},
        )
    axes[1].axvline(experimental_transition_K, color="#c4513e", ls=":", lw=1.6)
    axes[1].text(
        0.50,
        0.20,
        r"$T_{\mathrm{exp}}\simeq 600\,\mathrm{K}$",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=22,
        color="#9f3f31",
    )
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel(r"$G_{\delta}-G_{\alpha}$ (kJ mol$^{-1}$ f.u.$^{-1}$)")
    axes[1].set_ylim(bottom=-6.0)
    for axis in axes:
        axis.set_xlim(400.0, 650.0)
        axis.set_xticks(np.arange(400.0, 651.0, 50.0))
        axis.tick_params(direction="in", top=True, right=True, width=1.1, length=5)
    figure.savefig(results_directory / "phase_diagram.png", dpi=300)
    figure.savefig(results_directory / "phase_diagram.pdf")
    plt.close(figure)

    print("Wrote {}".format(results_directory / "phase_diagram.csv"))
    print("Wrote {}".format(results_directory / "phase_diagram.png"))
    if np.isfinite(crossing) and scientifically_interpretable:
        print("Estimated delta/cubic crossing: {:.2f} K".format(crossing))
    elif np.isfinite(crossing):
        print("Diagnostic crossing: {:.2f} K; quality checks failed".format(crossing))
    else:
        print("No delta/cubic crossing occurs inside the sampled temperature interval")
    if quality_issues:
        print("Quality issues:")
        for issue in quality_issues:
            print("- {}".format(issue))


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config, profile = load_configuration(config_path, args.profile)
    run_root = (
        args.run_directory.expanduser().resolve()
        if args.run_directory is not None
        else (SCRIPT_DIR / "runs" / args.profile).resolve()
    )
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if threads_per_job(args, config) < 1:
        raise ValueError("--threads-per-job must be positive")

    if args.stage in ("prepare", "all"):
        prepare(args, config, profile, run_root)
    if args.stage in ("npt", "all"):
        run_npt(args, config, profile, run_root)
    if args.stage in ("anchors", "all"):
        build_anchors(args, config, profile, run_root)
    if args.stage in ("msd", "all"):
        run_msd(args, config, profile, run_root)
    if args.stage in ("ti", "all"):
        run_ti(args, config, profile, run_root)
    if args.stage in ("analyze", "all") and not args.dry_run:
        analyze(args, config, profile, run_root)


if __name__ == "__main__":
    main()
