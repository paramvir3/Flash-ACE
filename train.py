import argparse
import yaml
import os
from typing import Optional
import torch
import torch.optim as optim
import numpy as np
import time
from ase.io import read
from ase.data import atomic_numbers, chemical_symbols
from e3nn import o3
from flashace.checkpoint import load_checkpoint
from flashace.model import TransformersACE, TransformersACEV3, TransformersACEV4
from flashace.optim import build_optimizer, optimizer_group_summary
from flashace.plotting import plot_metric_history
from ase.neighborlist import neighbor_list
from torch.utils.data import DataLoader, Dataset

# --- STABILITY SETTINGS ---
# Disable TF32 to prevent potential TensorCore precision crashes in e3nn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _make_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def configure_determinism(config) -> bool:
    """Optionally force bitwise-reproducible CUDA execution.

    TRACE scatters edge contributions onto atoms with ``index_add_``, whose CUDA
    implementation uses non-deterministic atomics: run-to-run energies differ by
    ~1e-5 meV/atom and gradients by ~1e-6 relative. That is far below the model
    error, but it makes a training run impossible to reproduce bitwise, which is
    what ``reproducibility.yaml`` promises. Enabling deterministic kernels costs
    about 7% and removes the variation entirely (measured on an RTX 5060 Ti).
    """
    if not bool(config.get('deterministic', False)):
        return False
    # cuBLAS needs this set before the first CUDA context to be reproducible.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    print("Deterministic mode: ON (CUBLAS_WORKSPACE_CONFIG=:4096:8, ~7% slower)")
    return True


def _load_config(config_arg: Optional[str]):
    """Load a YAML config, falling back to training/config.yaml if needed."""
    candidates = []
    if config_arg:
        candidates.append(config_arg)
    candidates.extend([
        "config.yaml",
        os.path.join("training", "config.yaml"),
    ])

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f), candidate

    raise FileNotFoundError(
        "No configuration file found. Provide --config or place config.yaml in the repo root or training/config.yaml."
    )


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, config, energy_shift_per_atom, atomic_energies):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'config': {
            'architecture_version': int(model.architecture_version),
            'r_max': config['r_max'],
            'l_max': config['l_max'],
            'num_radial': config['num_radial'],
            'hidden_dim': config['hidden_dim'],
            'num_layers': config['num_layers'],
            'radial_basis_type': config.get('radial_basis_type', 'bessel'),
            'radial_trainable': config.get('radial_trainable', False),
            'gaussian_width': config.get('gaussian_width', 0.5),
            'energy_shift_per_atom': (
                float(energy_shift_per_atom)
                if energy_shift_per_atom is not None
                else None
            ),
            'atomic_energies': atomic_energies or {},
            'amp_dtype': config.get('amp_dtype', 'float16'),
            'use_amp': config.get('use_amp', False),
            'grad_accum_steps': max(1, int(config.get('grad_accum_steps', 1))),
            'precompute_neighbors': config.get('precompute_neighbors', False),
            'radial_mlp_hidden': config.get('radial_mlp_hidden', 32),
            'radial_mlp_layers': config.get('radial_mlp_layers', 2),
            'correlation_order': config.get('correlation_order', 4),
            'correlation_channels': config.get('correlation_channels', 16),
            'attention_num_heads': config.get('attention_num_heads', config.get('transformer_num_heads', 4)),
            'attention_key_dim': config.get('attention_key_dim', None),
            'attention_ffn_hidden': config.get('attention_ffn_hidden', config.get('transformer_ffn_hidden', None)),
            'attention_dropout': config.get('attention_dropout', config.get('transformer_dropout', 0.0)),
            'attention_layer_scale_init': config.get('attention_layer_scale_init', 1e-2),
            'attention_distance_penalty': config.get('attention_distance_penalty', True),
            'attention_mode': config.get('attention_mode', 'full'),
            'attention_num_shells': config.get('attention_num_shells', 4),
            'correlation_rank_initial': config.get('correlation_rank_initial', None),
            'correlation_rank_warmup_epochs': config.get('correlation_rank_warmup_epochs', 0),
        }
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def build_neighbor_tensors(atoms, r_max):
    """Return neighbor-to-center edges plus periodic shifts from ASE."""
    i, j, shifts = neighbor_list('ijS', atoms, r_max)
    edge_index = torch.stack(
        [torch.tensor(j, dtype=torch.long), torch.tensor(i, dtype=torch.long)],
        dim=0,
    )
    edge_shift = torch.tensor(shifts, dtype=torch.float32)
    return edge_index, edge_shift


def _stress_matrix(value):
    """Convert an ASE/ExtXYZ stress-like value to a symmetric 3x3 matrix."""
    array = np.asarray(value, dtype=float)
    if array.shape == (6,):
        xx, yy, zz, yz, xz, xy = array
        array = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=float,
        )
    elif array.size == 9:
        array = array.reshape(3, 3)
    else:
        raise ValueError(
            f"Stress must contain 6 Voigt or 9 matrix components, got shape {array.shape}"
        )
    return 0.5 * (array + array.T)


def stress_target_from_atoms(atoms):
    """Return stress in eV/Angstrom^3 and whether a label was present."""
    results = atoms.calc.results if atoms.calc is not None else {}
    if 'stress' in results:
        return _stress_matrix(results['stress']), True
    if 'stress' in atoms.info:
        return _stress_matrix(atoms.info['stress']), True

    # The atomistic-data convention is virial = -V * stress. ASE ExtXYZ
    # normally performs this conversion while reading, but retain a correct
    # fallback for Atoms objects assembled by other readers.
    if 'virial' in results:
        return -_stress_matrix(results['virial']) / atoms.get_volume(), True
    if 'virial' in atoms.info:
        return -_stress_matrix(atoms.info['virial']) / atoms.get_volume(), True

    return np.zeros((3, 3), dtype=float), False


def stress_to_voigt(stress):
    """Return ASE-order Voigt components: xx, yy, zz, yz, xz, xy."""
    return torch.stack(
        [
            stress[0, 0], stress[1, 1], stress[2, 2],
            stress[1, 2], stress[0, 2], stress[0, 1],
        ]
    )


class AtomisticDataset(Dataset):
    def __init__(self, atoms_list, r_max, random_rotation=False, precompute_neighbors=False):
        self.atoms_list = atoms_list
        self.r_max = r_max
        self.random_rotation = random_rotation
        self.precompute_neighbors = precompute_neighbors

        self._edge_cache = None
        if precompute_neighbors:
            self._edge_cache = []
            for atoms in atoms_list:
                self._edge_cache.append(build_neighbor_tensors(atoms, self.r_max))
        
    def __len__(self): return len(self.atoms_list)
    
    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        
        # Geometry
        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        # Gas-phase molecular data normally have no simulation cell. Volume is
        # used only by the periodic stress path, so a neutral placeholder keeps
        # energy/force training well-defined without fabricating periodicity.
        volume = atoms.get_volume() if atoms.cell.rank == 3 else 1.0
        vol = torch.tensor(volume, dtype=torch.float32)
        
        # Targets
        t_E = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        t_F = torch.tensor(atoms.get_forces(), dtype=torch.float32)
        
        stress, has_stress = stress_target_from_atoms(atoms)
        t_S = torch.tensor(stress, dtype=torch.float32)
        
        # Neighbors
        if self.random_rotation:
            # Random SO(3) rotation applied via Wigner matrices to encourage rotationally
            # equivariant learning even on small datasets. Energies remain invariant
            # while forces/stresses are rotated consistently.
            rot = o3.rand_matrix().to(dtype=pos.dtype)
            pos = pos @ rot.T
            cell = cell @ rot.T
            t_F = t_F @ rot.T
            t_S = rot @ t_S @ rot.T

        if self._edge_cache is not None:
            edge_index, edge_shift = self._edge_cache[idx]
        else:
            edge_index, edge_shift = build_neighbor_tensors(atoms, self.r_max)

        return {
            'z': z,
            'pos': pos,
            'cell': cell,
            'edge_index': edge_index,
            'edge_shift': edge_shift,
            'volume': vol,
            't_E': t_E,
            't_F': t_F,
            't_S': t_S,
            'has_stress': torch.tensor(has_stress, dtype=torch.bool),
        }
    
    @staticmethod
    def collate_fn(batch): return batch

    @staticmethod
    def collate_batched(items):
        """Concatenate structures into ONE disconnected graph.

        Atom indices of structure b are offset by the running atom count, so the
        edge lists of different structures can never connect: the graph is a
        disjoint union and every per-atom quantity is unchanged. ``batch`` maps
        atom -> structure so the readout can be scattered back per structure.

        The GPU is launch-bound for single small cells, so one batched graph
        costs roughly the same as one structure while doing len(items) times the
        work. See docs/CUDA_NOTES.md.
        """
        z, pos, cells, vols, batch_idx = [], [], [], [], []
        e_idx, e_shift = [], []
        t_E, t_F, t_S, has_S = [], [], [], []
        offset = 0
        for b, it in enumerate(items):
            n = int(it['z'].shape[0])
            z.append(it['z']); pos.append(it['pos'])
            cells.append(it['cell']); vols.append(it['volume'].reshape(()))
            batch_idx.append(torch.full((n,), b, dtype=torch.long))
            e_idx.append(it['edge_index'] + offset)
            e_shift.append(it['edge_shift'])
            t_E.append(it['t_E'].reshape(())); t_F.append(it['t_F'])
            t_S.append(it['t_S']); has_S.append(it['has_stress'].reshape(()))
            offset += n
        return {
            'z': torch.cat(z), 'pos': torch.cat(pos),
            'cell': torch.stack(cells), 'volume': torch.stack(vols),
            'batch': torch.cat(batch_idx), 'n_structures': len(items),
            'edge_index': torch.cat(e_idx, dim=1), 'edge_shift': torch.cat(e_shift),
            't_E': torch.stack(t_E), 't_F': torch.cat(t_F),
            't_S': torch.stack(t_S), 'has_stress': torch.stack(has_S),
            'n_atoms': torch.tensor([int(it['z'].shape[0]) for it in items], dtype=torch.long),
        }

class MetricTracker:
    """Accumulates energy, force and stress errors.

    Besides RMSE_E (Eq. 42) this also tracks the *offset-free* energy error, the
    standard deviation of the per-atom energy error about its own mean.  RMSE_E
    is dominated by a global additive shift whenever the reference energy is a
    single scalar; the offset-free number isolates the part of the error that a
    constant reference cannot absorb, and is the quantity that is stable across
    epochs and library versions.
    """

    def __init__(self): self.reset()
    def reset(self):
        self.sse_e = 0.0; self.sse_s = 0.0
        # Weighted Welford accumulators for the offset-free energy error. The
        # naive <e^2> - <e>^2 form suffers catastrophic cancellation once the
        # energy zero drifts, which is exactly the regime this metric exists to
        # diagnose, so the streaming form is used instead.
        self._w_sum = 0.0
        self._mean_e = 0.0
        self._m2_e = 0.0
        self.sum_force_mse = 0.0
        self.sum_force_mae = 0.0
        self.n_atoms = 0; self.n_stress_comp = 0; self.n_struct = 0
    def update(self, p_E, p_F, p_S, t_E, t_F, t_S, include_stress, n_ats):
        # Promote to float64 *before* subtracting. Total energies are O(10^3 eV)
        # for a 40-atom CsPbI3 cell while the error is O(10^-1 eV); a float32
        # difference of two such numbers loses roughly three significant digits
        # of the quantity being measured.
        # ``.detach()`` first: p_E still carries the autograd graph here, and
        # calling float() on a requires_grad tensor warns and pins the graph alive.
        err_e = (p_E.detach().double().item() - torch.as_tensor(t_E).detach().double().item()) / n_ats
        self.sse_e += err_e**2 * n_ats
        weight = float(n_ats)
        self._w_sum += weight
        delta = err_e - self._mean_e
        self._mean_e += delta * weight / self._w_sum
        self._m2_e += weight * delta * (err_e - self._mean_e)
        diff_f = p_F - t_F
        # Per-structure force MSE/MAE averaged over 3N components.
        force_mse = diff_f.pow(2).mean().item()
        force_mae = diff_f.abs().mean().item()
        self.sum_force_mse += force_mse
        self.sum_force_mae += force_mae
        self.n_struct += 1
        if include_stress:
            self.sse_s += (
                stress_to_voigt(p_S) - stress_to_voigt(t_S)
            ).pow(2).sum().item()
            self.n_stress_comp += 6
        self.n_atoms += n_ats
    def update_batched(self, p_E, p_F, p_S, t_E, t_F, t_S, stress_mask, batch, n_atoms):
        """Accumulate a whole batch without a single host synchronisation.

        Every ``.item()`` forces the CPU to wait for the GPU. With one structure
        per call that was ~4 syncs x |B| per step; here the batch statistics are
        reduced on device and merged into the running totals with Chan's
        parallel variance formula, which is algebraically identical to the
        sequential Welford update but needs no per-structure round trip. The
        only synchronisation is in ``get_metrics`` once per epoch.
        """
        n = n_atoms.double()
        err_e = (p_E.detach().double() - t_E.detach().double()) / n          # [B] eV/atom
        self.sse_e = self.sse_e + (err_e ** 2 * n).sum()

        # ---- Chan et al. parallel merge of (weight, mean, M2) ----
        w_b = n.sum()
        mean_b = (err_e * n).sum() / w_b
        m2_b = (n * (err_e - mean_b) ** 2).sum()
        w_a = self._w_sum
        delta = mean_b - self._mean_e
        w_new = w_a + w_b
        self._mean_e = self._mean_e + delta * w_b / w_new
        self._m2_e = self._m2_e + m2_b + delta ** 2 * w_a * w_b / w_new
        self._w_sum = w_new

        diff_f = (p_F.detach() - t_F.detach()).double()                       # [N_tot, 3]
        sq = torch.zeros_like(err_e).index_add(0, batch, diff_f.pow(2).sum(dim=1))
        ab = torch.zeros_like(err_e).index_add(0, batch, diff_f.abs().sum(dim=1))
        self.sum_force_mse = self.sum_force_mse + (sq / (3.0 * n)).sum()
        self.sum_force_mae = self.sum_force_mae + (ab / (3.0 * n)).sum()

        mask = stress_mask.detach().double()
        if float(mask.sum()) > 0.0:
            d2 = ((p_S.detach() - t_S.detach()).double() ** 2).flatten(1)
            voigt = d2[:, [0, 4, 8, 5, 2, 1]].sum(dim=1)                      # xx yy zz yz xz xy
            self.sse_s = self.sse_s + (mask * voigt).sum()
            self.n_stress_comp += int(6 * mask.sum().item())

        self.n_struct += int(p_E.shape[0])
        self.n_atoms = self.n_atoms + n.sum()

    def _as_float(self, x):
        return float(x) if not torch.is_tensor(x) else float(x.detach().cpu())

    def get_metrics(self):
        """Return metrics. This is the ONLY host synchronisation point."""
        sse_e   = self._as_float(self.sse_e)
        sse_s   = self._as_float(self.sse_s)
        n_atoms = self._as_float(self.n_atoms)
        sum_mse = self._as_float(self.sum_force_mse)
        sum_mae = self._as_float(self.sum_force_mae)
        w_sum   = self._as_float(self._w_sum)
        mean_e  = self._as_float(self._mean_e)
        m2_e    = self._as_float(self._m2_e)

        rmse_e = np.sqrt(sse_e / n_atoms) if n_atoms > 0 else 0.0
        rmse_s = np.sqrt(sse_s / self.n_stress_comp) if self.n_stress_comp > 0 else 0.0
        force_mse = (sum_mse / self.n_struct) if self.n_struct > 0 else 0.0
        force_mae = (sum_mae / self.n_struct) if self.n_struct > 0 else 0.0
        force_rmse = np.sqrt(force_mse)
        # Offset-free energy error: the N_s-weighted standard deviation of the
        # per-atom error about its own mean. Exactly invariant under a constant
        # shift of the energy reference, unlike rmse_e.
        if w_sum > 0.0:
            rmse_e_free = np.sqrt(max(m2_e / w_sum, 0.0))
        else:
            mean_e = 0.0
            rmse_e_free = 0.0
        return (
            rmse_e * 1000, force_rmse, rmse_s, force_mse, force_mae,
            rmse_e_free * 1000, mean_e * 1000,
        )


def compute_mean_energy_per_atom(atoms_seq):
    total_energy = 0.0
    total_atoms = 0
    for atoms in atoms_seq:
        total_energy += atoms.get_potential_energy()
        total_atoms += len(atoms)
    return (total_energy / total_atoms) if total_atoms > 0 else 0.0


def compute_atomic_energies_from_dataset(atoms_seq):
    """Solve for per-species reference energies via least squares.

    Builds the standard per-species linear system where each structure's
    total energy is expressed as the sum of per-species reference energies plus
    a residual. The least-squares solution provides offsets that remove most of
    the composition-dependent baseline from the supervised loss.
    """

    species = sorted({int(z) for atoms in atoms_seq for z in atoms.numbers})
    if not species:
        return {}

    counts = []
    energies = []
    for atoms in atoms_seq:
        counts.append([np.count_nonzero(atoms.numbers == z) for z in species])
        energies.append(atoms.get_potential_energy())

    X = np.array(counts, dtype=float)
    y = np.array(energies, dtype=float)

    coeffs, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)

    # With a fixed stoichiometry (every frame Cs8Pb8I24, say) the composition
    # matrix is rank deficient and the individual per-species energies are not
    # identifiable -- only their composition-weighted sum is. lstsq silently
    # returns the minimum-norm solution, which splits the degenerate species
    # equally. That baseline is still exactly right for every training frame, but
    # the individual numbers are arbitrary and must not be transferred to a
    # different composition.
    if rank < len(species):
        symbols = ", ".join(chemical_symbols[z] for z in species)
        print(
            f"WARNING: the composition matrix has rank {rank} < {len(species)} species "
            f"({symbols}). Per-species reference energies are not separately "
            "identifiable from this dataset; the minimum-norm solution is used. The "
            "total reference per structure is correct, but do not reuse these "
            "per-species values for a different composition."
        )
    return {z: float(e) for z, e in zip(species, coeffs)}


def parse_atomic_energy_table(raw_table):
    """Normalize an atomic energy mapping with atomic numbers as keys."""

    table = {}
    if raw_table is None:
        return table

    for key, value in raw_table.items():
        if isinstance(key, str):
            try:
                z = atomic_numbers[key]
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Unknown chemical symbol '{key}' in atomic_energies") from exc
        else:
            z = int(key)

        table[z] = float(value)

    return table


def atomic_energy_tensor(energy_table, device):
    if not energy_table:
        return None

    max_z = max(energy_table)
    tensor = torch.zeros(max_z + 1, dtype=torch.float32, device=device)
    for z, val in energy_table.items():
        tensor[z] = val
    return tensor


def split_trajectory_frames(
    atoms,
    val_fraction=0.1,
    seed=42,
    mode="blocked",
    block_size=25,
    gap=0,
):
    """Split ordered trajectory frames without scattering one trajectory block.

    A frame-random split gives unrealistically optimistic validation estimates
    when adjacent molecular-dynamics frames are strongly correlated. ``blocked``
    selects whole contiguous blocks and can exclude a small boundary gap from
    training. An explicit validation file remains preferable when phase or
    trajectory labels are available.
    """
    n_frames = len(atoms)
    if n_frames < 2:
        raise ValueError("At least two structures are required for a train/validation split")

    val_target = max(1, min(n_frames - 1, int(round(n_frames * val_fraction))))
    generator = torch.Generator().manual_seed(int(seed))
    mode = str(mode).lower()

    if mode == "random":
        order = torch.randperm(n_frames, generator=generator).tolist()
        val_indices = sorted(order[:val_target])
    elif mode == "blocked":
        block_size = min(max(1, int(block_size)), max(1, n_frames // 2))
        blocks = [
            list(range(start, min(start + block_size, n_frames)))
            for start in range(0, n_frames, block_size)
        ]
        order = torch.randperm(len(blocks), generator=generator).tolist()
        selected = []
        selected_count = 0
        for block_index in order:
            if selected_count >= val_target:
                break
            if len(blocks) - len(selected) <= 1:
                break
            selected.append(block_index)
            selected_count += len(blocks[block_index])
        val_indices = sorted(i for block_index in selected for i in blocks[block_index])
    else:
        raise ValueError("split_mode must be 'blocked' or 'random'")

    val_set = set(val_indices)
    excluded = set()
    gap = max(0, int(gap))
    if gap:
        for index in val_indices:
            excluded.update(range(max(0, index - gap), min(n_frames, index + gap + 1)))
        excluded.difference_update(val_set)

    train_indices = [i for i in range(n_frames) if i not in val_set and i not in excluded]
    if not train_indices or not val_indices:
        raise ValueError("Split settings left an empty training or validation set")

    return (
        [atoms[i] for i in train_indices],
        [atoms[i] for i in val_indices],
        sorted(excluded),
    )


def _batched_train_step(
    model, items, device, temp_scale,
    energy_weight, force_weight, stress_weight,
    sobolev_weight, sobolev_sigma,
    baseline_energy, metrics, scaler, norm,
):
    """One optimiser sub-step over a batch evaluated as a single graph.

    Reproduces Eqs. (37)-(41) exactly, but vectorised over structures. The
    per-structure normalisations matter: energies are divided by N_s before
    squaring, force errors averaged over 3N_s, stress over its six Voigt
    components, and stress-free structures contribute zero while still counting
    in the batch denominator.
    """
    bat = AtomisticDataset.collate_batched(items)
    for k, v in bat.items():
        if isinstance(v, torch.Tensor):
            bat[k] = v.to(device, non_blocking=True)

    n_at = bat['n_atoms'].to(device).to(bat['pos'].dtype)          # [B]
    has_s = bat['has_stress'].to(device).to(bat['pos'].dtype)      # [B]
    want_stress = bool(has_s.any().item()) and stress_weight > 0.0

    sob_active = sobolev_weight > 0.0 and sobolev_sigma > 0.0
    rng_cpu = torch.get_rng_state() if sob_active else None
    rng_dev = (
        torch.cuda.get_rng_state(device)
        if sob_active and torch.device(device).type == 'cuda' else None
    )

    E, F, S, _ = model.forward_batched(
        bat, training=True, temperature_scale=temp_scale, compute_stress=want_stress
    )

    base = torch.stack([baseline_energy(it['z'].to(device)) for it in items]).reshape(-1)
    target_E = bat['t_E'].reshape(-1) - base
    loss_e = (((E - target_E) / n_at) ** 2).sum()

    dF2 = (F - bat['t_F']) ** 2                                    # [N_total, 3]
    per_struct_f = torch.zeros_like(E).index_add(0, bat['batch'], dF2.sum(dim=1))
    loss_f = (per_struct_f / (3.0 * n_at)).sum()

    loss = energy_weight * loss_e + force_weight * loss_f
    if want_stress:
        dS2 = ((S - bat['t_S']) ** 2).flatten(1)                   # [B, 9]
        voigt = dS2[:, [0, 4, 8, 5, 2, 1]].sum(dim=1)              # xx yy zz yz xz xy
        loss = loss + stress_weight * (has_s * voigt / 6.0).sum()

    if sob_active:
        delta = torch.randn_like(bat['pos']) * sobolev_sigma
        pert = dict(bat); pert['pos'] = bat['pos'] + delta
        torch.set_rng_state(rng_cpu)
        if rng_dev is not None:
            torch.cuda.set_rng_state(rng_dev, device)
        E_p, _, _, _ = model.forward_batched(
            pert, training=True, temperature_scale=temp_scale,
            detach_pos=True, compute_stress=False,
        )
        fd_lin = torch.zeros_like(E).index_add(0, bat['batch'], (F.detach() * delta).sum(dim=1))
        loss = loss + sobolev_weight * (((E_p - E) + fd_lin) ** 2).sum()

    scaler.scale(loss / norm).backward()

    with torch.no_grad():
        S_all = S.detach() if want_stress else torch.zeros(
            len(items), 3, 3, device=device, dtype=bat['pos'].dtype
        )
        mask = has_s if want_stress else torch.zeros_like(has_s)
        metrics.update_batched(
            E, F, S_all, target_E, bat['t_F'], bat['t_S'],
            mask, bat['batch'], bat['n_atoms'].to(device),
        )
    return loss.detach()


def main():
    parser = argparse.ArgumentParser(description="Train Transformers-ACE")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    args = parser.parse_args()

    config, config_path = _load_config(args.config)
    configure_determinism(config)
    print(f"--- Loading {config_path} ---")

    seed = int(config.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_num_threads = int(config.get('torch_num_threads', 0) or 0)
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    torch_num_interop_threads = int(config.get('torch_num_interop_threads', 0) or 0)
    if torch_num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(torch_num_interop_threads)
        except RuntimeError:
            # PyTorch only permits changing this before inter-op work starts.
            pass
    print(
        f"PyTorch CPU threads: {torch.get_num_threads()} intra-op, "
        f"{torch.get_num_interop_threads()} inter-op"
    )

    device = config['device']
    device_type = device.split(":")[0]

    use_amp = config.get('use_amp', False) and device_type == 'cuda'
    amp_dtype = torch.float16 if config.get('amp_dtype', 'float16') == 'float16' else torch.bfloat16
    grad_accum_steps = max(1, int(config.get('grad_accum_steps', 1)))

    if device_type == "cuda":
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    

    print(f"Reading data from {config['train_file']}...")
    all_atoms = read(config['train_file'], index=":")
    stress_labels = sum(stress_target_from_atoms(atoms)[1] for atoms in all_atoms)
    print(f"Stress labels: {stress_labels}/{len(all_atoms)} structures")
    configured_stress_weight = max(
        float(config.get('stress_weight', 0.0)),
        float(config.get('stress_weight_final', config.get('stress_weight', 0.0))),
    )
    if configured_stress_weight > 0.0 and stress_labels == 0:
        raise ValueError("stress_weight is nonzero, but the dataset contains no stress labels")

    if config['valid_file']:
        val_atoms = read(config['valid_file'], index=":")
        train_atoms = all_atoms
    else:
        split_mode = config.get('split_mode', 'blocked')
        train_atoms, val_atoms, excluded_frames = split_trajectory_frames(
            all_atoms,
            val_fraction=config.get('val_split', 0.1),
            seed=seed,
            mode=split_mode,
            block_size=config.get('split_block_size', 25),
            gap=config.get('split_gap', 0),
        )
        print(
            f"{str(split_mode).title()} split: {len(train_atoms)} Training | "
            f"{len(val_atoms)} Validation | {len(excluded_frames)} Gap-excluded"
        )

    atomic_energy_map = parse_atomic_energy_table(config.get('atomic_energies'))

    if not atomic_energy_map and config.get('solve_atomic_energies', False):
        atomic_energy_map = compute_atomic_energies_from_dataset(train_atoms)
        if atomic_energy_map:
            pretty = ", ".join(
                f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
            )
            print(f"Solved per-species reference energies from training set -> {pretty}")

    if atomic_energy_map:
        energy_shift_per_atom = None
        config['atomic_energies'] = atomic_energy_map
        pretty = ", ".join(
            f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
        )
        print(f"Using per-species reference energies for normalization: {pretty}")
    elif config.get('energy_shift_per_atom') is not None:
        energy_shift_per_atom = float(config['energy_shift_per_atom'])
        print(f"Using user-provided energy shift per atom: {energy_shift_per_atom:.6f} eV")
    else:
        energy_shift_per_atom = compute_mean_energy_per_atom(train_atoms)
        print(f"Computed mean energy per atom for normalization: {energy_shift_per_atom:.6f} eV")

    # DATALOADERS
    train_ds = AtomisticDataset(
        train_atoms,
        config['r_max'],
        random_rotation=config.get('random_rotation', False),
        precompute_neighbors=config.get('precompute_neighbors', False),
    )
    val_ds = AtomisticDataset(
        val_atoms,
        config['r_max'],
        random_rotation=False,
        precompute_neighbors=config.get('precompute_neighbors', False),
    )

    num_workers = int(config.get('num_workers', 2))
    pin_memory = device_type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              generator=torch.Generator().manual_seed(seed))
    
    valid_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, num_workers=num_workers)

    print("--- Initializing Transformers-ACE ---")
    architecture_version = int(config.get('architecture_version', 2))
    if architecture_version == 2:
        model_class = TransformersACE
    elif architecture_version == 3:
        model_class = TransformersACEV3
    elif architecture_version == 4:
        model_class = TransformersACEV4
    else:
        raise ValueError("architecture_version must be 2, 3, or 4 for training")

    model_kwargs = dict(
        r_max=config['r_max'], l_max=config['l_max'], num_radial=config['num_radial'],
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
        radial_basis_type=config.get('radial_basis_type', 'bessel'),
        radial_trainable=config.get('radial_trainable', False),
        gaussian_width=config.get('gaussian_width', 0.5),
        attention_num_heads=config.get('attention_num_heads', config.get('transformer_num_heads', 4)),
        attention_key_dim=config.get('attention_key_dim', None),
        attention_ffn_hidden=config.get('attention_ffn_hidden', config.get('transformer_ffn_hidden', None)),
        attention_dropout=config.get('attention_dropout', config.get('transformer_dropout', 0.0)),
        attention_layer_scale_init=config.get('attention_layer_scale_init', 1e-2),
        attention_distance_penalty=config.get('attention_distance_penalty', True),
        attention_mode=config.get('attention_mode', 'full'),
        radial_mlp_hidden=config.get('radial_mlp_hidden', 32),
        radial_mlp_layers=config.get('radial_mlp_layers', 2),
        correlation_order=config.get('correlation_order', 4),
        correlation_channels=config.get('correlation_channels', 16),
        use_aux_force_head=False,
        use_aux_stress_head=False,
    )
    if architecture_version == 4:
        model_kwargs.update(
            attention_num_shells=config.get('attention_num_shells', 4),
            correlation_rank_initial=config.get('correlation_rank_initial', None),
        )
    model = model_class(**model_kwargs).to(device)
    
    optimizer = build_optimizer(model, config)
    print(f"Optimizer: {config.get('optimizer', 'adam')}")
    print(f"Optimizer groups: {optimizer_group_summary(optimizer)}")
    warmup_epochs = max(0, int(config.get('lr_warmup_epochs', 0)))
    warmup_start = float(config.get('lr_warmup_start_factor', 0.1))
    scheduler_interval = str(config.get('lr_scheduler_interval', 'epoch')).lower()
    if warmup_start <= 0.0:
        raise ValueError("lr_warmup_start_factor must be > 0.0")
    if scheduler_interval not in {'epoch', 'step'}:
        raise ValueError("lr_scheduler_interval must be 'epoch' or 'step'")

    steps_per_epoch = max(1, len(train_loader))
    configured_t_max = int(config.get('lr_scheduler_t_max', config['epochs']))
    if scheduler_interval == 'step':
        warmup_iters = warmup_epochs * steps_per_epoch
        total_iters = max(warmup_iters + 1, configured_t_max * steps_per_epoch)
    else:
        warmup_iters = warmup_epochs
        total_iters = max(warmup_epochs + 1, configured_t_max)
    cosine_t_max = max(1, total_iters - warmup_iters)

    use_restarts = bool(config.get('lr_scheduler_use_restarts', False))
    restart_mult = float(config.get('lr_restart_mult', 1.0))
    if use_restarts:
        cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_t_max,
            T_mult=max(1.0, restart_mult),
            eta_min=config.get('lr_min', 0.0),
        )
    else:
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=config.get('lr_min', 0.0),
        )
    if warmup_iters > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start,
            total_iters=warmup_iters,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_iters],
        )
    else:
        scheduler = cosine

    # Optional force-weight annealing to ease optimization toward energies.
    force_w_start = float(config.get('forces_weight', 10.0))
    force_w_final = float(config.get('forces_weight_final', force_w_start))
    force_w_decay_epochs = int(config.get('forces_weight_decay_epochs', 0))

    def _force_weight(epoch_idx: int) -> float:
        if force_w_decay_epochs <= 0 or force_w_start == force_w_final:
            return force_w_start
        frac = min(1.0, epoch_idx / max(1, force_w_decay_epochs))
        return force_w_start + frac * (force_w_final - force_w_start)

    stress_w_start = float(config.get('stress_weight', 0.0))
    stress_w_final = float(config.get('stress_weight_final', stress_w_start))
    stress_w_ramp_epochs = int(config.get('stress_weight_ramp_epochs', 0))

    def _stress_weight(epoch_idx: int) -> float:
        if stress_w_ramp_epochs <= 0 or stress_w_start == stress_w_final:
            return stress_w_start
        frac = min(1.0, epoch_idx / max(1, stress_w_ramp_epochs))
        return stress_w_start + frac * (stress_w_final - stress_w_start)

    resume_path = config.get('resume_from')
    start_epoch = 0

    scaler = _make_grad_scaler(use_amp)

    if resume_path:
        print(f"--- Loading checkpoint from {resume_path} ---")
        checkpoint = load_checkpoint(resume_path, map_location=device)
        checkpoint_version = int(checkpoint.get('config', {}).get('architecture_version', 1))
        if checkpoint_version != architecture_version:
            raise ValueError(
                f"Cannot resume architecture v{checkpoint_version} weights in the "
                f"v{architecture_version} model. Start a matching architecture run; "
                "the calculator can still evaluate older checkpoints."
            )
        model.load_state_dict(checkpoint['model_state_dict'])

        if config.get('resume_load_optimizer', False) and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if config.get('resume_load_scheduler', False) and checkpoint.get('scheduler_state_dict'):
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if config.get('resume_load_scaler', False) and checkpoint.get('scaler_state_dict') and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if config.get('use_checkpoint_energy_shift', True):
            ckpt_atomic = checkpoint.get('config', {}).get('atomic_energies') or {}
            if ckpt_atomic:
                atomic_energy_map = {int(k): float(v) for k, v in ckpt_atomic.items()}
                energy_shift_per_atom = None
                config['atomic_energies'] = atomic_energy_map
                pretty = ", ".join(
                    f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
                )
                print(f"Using checkpoint atomic energy references: {pretty}")
            else:
                ckpt_shift = checkpoint.get('config', {}).get('energy_shift_per_atom')
                if ckpt_shift is not None:
                    energy_shift_per_atom = ckpt_shift
                    print(f"Using checkpoint energy shift per atom: {energy_shift_per_atom:.6f} eV")

        start_epoch = int(checkpoint.get('epoch', 0))
        print(f"Resuming training from epoch {start_epoch}")

    energy_shift = None
    if energy_shift_per_atom is not None:
        energy_shift = torch.tensor(energy_shift_per_atom, dtype=torch.float32, device=device)

    atomic_energy_vec = atomic_energy_tensor(atomic_energy_map, device)

    def baseline_energy(z_tensor):
        if atomic_energy_vec is not None:
            if torch.max(z_tensor).item() >= atomic_energy_vec.shape[0]:
                raise ValueError("Encountered atomic number without reference energy")
            return torch.sum(atomic_energy_vec[z_tensor])
        elif energy_shift is not None:
            return energy_shift * len(z_tensor)
        else:
            return torch.tensor(0.0, device=device)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_energy_rmse': [],
        'val_energy_rmse': [],
        'train_energy_rmse_offset_free': [],
        'val_energy_rmse_offset_free': [],
        'train_energy_bias': [],
        'val_energy_bias': [],
        'train_force_rmse': [],
        'val_force_rmse': [],
        'train_stress_rmse': [],
        'val_stress_rmse': [],
    }

    ckpt_interval = int(config.get('checkpoint_interval', 0) or 0)

    # Attention-temperature curriculum.
    #
    # T_att must be a deterministic function of the epoch alone and must reach
    # exactly 1 before training ends, because validation, the ASE calculator and
    # every deployed path evaluate the model at T_att = 1.  Any residual
    # T_att != 1 at the final epoch means the optimizer is shaping a *different*
    # function from the one that is later deployed.
    temp_scale_start = float(config.get('temperature_scale_start', 1.0))
    temp_scale_end = float(config.get('temperature_scale_end', 1.0))
    temp_scale_epochs = int(config.get('temperature_scale_epochs', 0) or 0)

    if abs(temp_scale_end - 1.0) > 1e-12:
        raise ValueError(
            "temperature_scale_end must be 1.0: validation and inference evaluate the "
            f"model at T_att = 1, but the schedule ends at {temp_scale_end}. Training "
            "would optimize a different function from the one deployed."
        )
    if temp_scale_epochs > 0 and temp_scale_epochs >= int(config['epochs']):
        raise ValueError(
            f"temperature_scale_epochs ({temp_scale_epochs}) must be smaller than "
            f"epochs ({int(config['epochs'])}) so that T_att reaches 1 before training ends."
        )
    # The force-loss feedback term made T_att data-dependent and left it at
    # ~4.4 at convergence while validation used 1.0.  It is no longer supported.
    for _legacy_key in ('temperature_force_ref', 'temperature_force_exponent'):
        if float(config.get(_legacy_key, 0.0) or 0.0) != 0.0:
            raise ValueError(
                f"'{_legacy_key}' is no longer supported: it made the attention "
                "temperature depend on the running force loss, so training and "
                "inference used different temperatures. Remove it from the config "
                "(the epoch schedule already anneals T_att to 1)."
            )

    def _temperature_scale(epoch_idx: int) -> float:
        """T_att as a pure function of epoch, annealing to exactly 1."""
        if temp_scale_epochs <= 0:
            return 1.0
        frac = min(1.0, epoch_idx / float(temp_scale_epochs))
        return temp_scale_start + frac * (temp_scale_end - temp_scale_start)

    # Evaluate a whole batch as one disconnected graph. The GPU is launch-bound
    # for single small cells, so this is a large throughput win; it is
    # numerically identical to the per-structure path (tests/test_batched_graph.py).
    batched_graphs = bool(config.get('batched_graphs', False))
    force_consistency_weight = float(config.get('force_consistency_weight', 0.0))
    if batched_graphs and force_consistency_weight > 0.0:
        raise ValueError(
            "batched_graphs does not support force_consistency_weight; the batched "
            "path always differentiates the batched position tensor."
        )
    displacement_prob = float(config.get('displacement_prob', 0.0))
    displacement_sigma = float(config.get('displacement_sigma', 0.0))
    if displacement_prob > 0.0:
        raise ValueError(
            "displacement augmentation cannot reuse the original DFT labels; "
            "set displacement_prob and displacement_sigma to 0"
        )
    aux_force_weight = float(config.get('aux_force_weight', 0.0))
    aux_stress_weight = float(config.get('aux_stress_weight', 0.0))
    sobolev_weight = float(config.get('sobolev_weight', 0.0))
    sobolev_sigma = float(config.get('sobolev_sigma', 0.0))

    early_stopping_patience = max(0, int(config.get('early_stopping_patience', 0)))
    early_stopping_min_epoch = max(
        stress_w_ramp_epochs,
        int(config.get('early_stopping_min_epoch', stress_w_ramp_epochs)),
    )
    early_stopping_min_delta = float(config.get('early_stopping_min_delta', 0.0))
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint_saved = False
    epochs_completed = start_epoch
    
    print(
        f"{'Epoch':>5} | {'Loss':>10} | {'E (meV)':>10} | {'E_free':>9} | {'force_RMSE':>12} | {'force_MSE':>12} | {'force_MAE':>12} | {'S_RMSE':>10} || "
        f"{'Val Loss':>10} | {'Val E':>10} | {'Val E_free':>10} | {'Val force_RMSE':>16} | {'Val force_MSE':>16} | {'Val S_RMSE':>12}"
    )
    print("-" * 170)
    
    force_loss_ema = None
    for epoch in range(start_epoch, config['epochs']):
        if architecture_version == 4:
            max_rank = int(model.ace.irreps_correlation[0].mul)
            initial_rank = int(config.get('correlation_rank_initial') or max_rank)
            rank_warmup = max(0, int(config.get('correlation_rank_warmup_epochs', 0)))
            if rank_warmup > 0:
                fraction = min(1.0, float(epoch + 1) / float(rank_warmup))
                active_rank = round(initial_rank + fraction * (max_rank - initial_rank))
            else:
                active_rank = max_rank
            model.set_correlation_rank(active_rank)
        force_weight = _force_weight(epoch)
        stress_weight = _stress_weight(epoch)
        model.train()
        train_metrics = MetricTracker()
        total_loss = 0.0
        total_items_seen = 0

        batch_idx = -1
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            batch_loss = 0.0

            # --- GRADIENT ACCUMULATION (FP32/AMP) ---
            items = list(batch)

            if batched_graphs:
                batch_loss = _batched_train_step(
                    model, items, device, _temperature_scale(epoch),
                    config['energy_weight'], force_weight, stress_weight,
                    sobolev_weight, sobolev_sigma,
                    baseline_energy, train_metrics, scaler,
                    len(items) * grad_accum_steps,
                )
                total_items_seen += len(items)
                if (batch_idx + 1) % grad_accum_steps == 0:
                    if config.get('clip_grad_norm', None):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler_interval == 'step':
                        scheduler.step()
                total_loss = total_loss + batch_loss
                continue

            for item in items:
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        item[k] = v.to(device, non_blocking=True)
                if force_consistency_weight > 0.0:
                    item['pos'] = item['pos'].clone().detach().requires_grad_(True)

                # Standard Forward with optional AMP
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    temp_scale = _temperature_scale(epoch)
                    stress_target = (
                        bool(item['has_stress'].item())
                        and stress_weight > 0.0
                    )
                    # The local-linearisation term (Eq. 41) compares two forward
                    # passes.  With independent attention-dropout masks their
                    # difference is dominated by dropout variance rather than by
                    # curvature, so the RNG state is captured here and replayed for
                    # the second pass, giving both passes identical masks.
                    sobolev_active = sobolev_weight > 0.0 and sobolev_sigma > 0.0
                    rng_state_cpu = torch.get_rng_state() if sobolev_active else None
                    rng_state_dev = (
                        torch.cuda.get_rng_state(device)
                        if sobolev_active and device_type == 'cuda'
                        else None
                    )
                    p_E, p_F, p_S, aux = model(
                        item,
                        training=True,
                        temperature_scale=temp_scale,
                        detach_pos=force_consistency_weight <= 0.0,
                        compute_stress=stress_target,
                    )
                    n_ats = len(item['z'])

                    target_E = item['t_E'] - baseline_energy(item['z'])
                    loss_e = ((p_E - target_E) / n_ats)**2
                    loss_f = torch.mean((p_F - item['t_F'])**2)
                    loss_s = torch.tensor(0.0, device=device)
                    if stress_target:
                        loss_s = torch.mean(
                            (stress_to_voigt(p_S) - stress_to_voigt(item['t_S']))**2
                        )

                    loss_item = (config['energy_weight'] * loss_e) + \
                                (force_weight * loss_f) + \
                                (stress_weight * loss_s)

                    if aux_force_weight > 0.0 and 'force' in aux:
                        loss_item = loss_item + aux_force_weight * torch.mean((aux['force'] - item['t_F'])**2)
                    if aux_stress_weight > 0.0 and 'stress' in aux:
                        target_stress = item['t_S']
                        loss_item = loss_item + aux_stress_weight * torch.mean(
                            (aux['stress'] - torch.stack([
                                target_stress[0,0], target_stress[1,1], target_stress[2,2],
                                target_stress[0,1], target_stress[0,2], target_stress[1,2]
                            ]))**2
                        )

                    if sobolev_active:
                        # Draw the displacement first, then rewind the RNG so the
                        # perturbed pass replays exactly the dropout masks used above.
                        delta = torch.randn_like(item['pos']) * sobolev_sigma
                        pos_pert = item['pos'] + delta
                        perturbed = {**item, 'pos': pos_pert}
                        torch.set_rng_state(rng_state_cpu)
                        if rng_state_dev is not None:
                            torch.cuda.set_rng_state(rng_state_dev, device)
                        p_E_pert, _, _, _ = model(
                            perturbed,
                            training=True,
                            temperature_scale=temp_scale,
                            detach_pos=True,
                            compute_stress=False,
                        )
                        # E(r+d) - E(r) + F.d  vanishes to first order; with matched
                        # masks this measures curvature, not dropout variance.
                        fd = (p_E_pert - p_E) + (p_F.detach() * delta).sum()
                        loss_item = loss_item + sobolev_weight * fd.pow(2)

                    if force_consistency_weight > 0.0:
                        energy_grad = torch.autograd.grad(
                            p_E,
                            item['pos'],
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if energy_grad is not None:
                            consistency = (p_F + energy_grad).pow(2).mean()
                            loss_item = loss_item + force_consistency_weight * consistency

                # Normalize and Backward
                loss_batch = loss_item / (len(items) * grad_accum_steps)
                scaler.scale(loss_batch).backward()

                batch_loss += loss_item.item()

                with torch.no_grad():
                    # Pass the *shifted* energies. The composition baseline cancels
                    # in the difference, so this is analytically identical to
                    # comparing absolute energies but avoids differencing two
                    # O(10^3 eV) float32 numbers to extract an O(10^-1 eV) error.
                    train_metrics.update(
                        p_E, p_F, p_S, target_E, item['t_F'], item['t_S'],
                        stress_target, n_ats,
                    )
                    if force_loss_ema is None:
                        force_loss_ema = loss_f.detach()
                    else:
                        force_loss_ema = 0.9 * force_loss_ema + 0.1 * loss_f.detach()
                total_items_seen += 1

            if (batch_idx + 1) % grad_accum_steps == 0:
                if config.get('clip_grad_norm', None):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['clip_grad_norm']
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler_interval == 'step':
                    scheduler.step()
            total_loss += batch_loss

        # Flush any residual gradients if the last batch didn't trigger a step
        if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
            if config.get('clip_grad_norm', None):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['clip_grad_norm']
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler_interval == 'step':
                scheduler.step()

        avg_train_loss = total_loss / max(1, total_items_seen)
        if torch.is_tensor(avg_train_loss):
            avg_train_loss = float(avg_train_loss.detach().cpu())
        tr_e, tr_f, tr_s, tr_f_mse, tr_f_mae, tr_e_free, tr_e_bias = train_metrics.get_metrics()
        # Validation
        model.eval()
        val_metrics = MetricTracker()
        val_loss_accum = 0.0

        for batch in valid_loader:
            if batched_graphs:
                items = list(batch)
                bat = AtomisticDataset.collate_batched(items)
                for k, v in bat.items():
                    if isinstance(v, torch.Tensor):
                        bat[k] = v.to(device, non_blocking=True)
                n_at = bat['n_atoms'].to(device).to(bat['pos'].dtype)
                has_s = bat['has_stress'].to(device).to(bat['pos'].dtype)
                want_s = bool(has_s.any().item()) and stress_weight > 0.0
                # Validation and deployment both evaluate at T_att = 1 (D2).
                p_E, p_F, p_S, _ = model.forward_batched(
                    bat, training=False, compute_stress=want_s
                )
                base = torch.stack(
                    [baseline_energy(it['z'].to(device)) for it in items]
                ).reshape(-1)
                target_E = bat['t_E'].reshape(-1) - base
                loss_e = (((p_E - target_E) / n_at) ** 2).sum()
                dF2 = (p_F - bat['t_F']) ** 2
                per_f = torch.zeros_like(p_E).index_add(0, bat['batch'], dF2.sum(dim=1))
                loss_f = (per_f / (3.0 * n_at)).sum()
                vloss = config['energy_weight'] * loss_e + force_weight * loss_f
                if want_s:
                    dS2 = ((p_S - bat['t_S']) ** 2).flatten(1)
                    voigt = dS2[:, [0, 4, 8, 5, 2, 1]].sum(dim=1)
                    vloss = vloss + stress_weight * (has_s * voigt / 6.0).sum()
                val_loss_accum += vloss.detach()
                S_all = p_S.detach() if want_s else torch.zeros(
                    len(items), 3, 3, device=device, dtype=bat['pos'].dtype
                )
                val_metrics.update_batched(
                    p_E, p_F, S_all, target_E, bat['t_F'], bat['t_S'],
                    has_s if want_s else torch.zeros_like(has_s),
                    bat['batch'], bat['n_atoms'].to(device),
                )
                continue
            for item in batch:
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        item[k] = v.to(device, non_blocking=True)

                # Keep grad tracking on so autograd can form forces/stresses; we
                # still avoid higher-order graphs with ``create_graph=False``
                # inside the model during validation.
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    stress_target = (
                        bool(item['has_stress'].item())
                        and stress_weight > 0.0
                    )
                    p_E, p_F, p_S, _ = model(
                        item,
                        training=False,
                        compute_stress=stress_target,
                    )
                    n_ats = len(item['z'])
                    target_E = item['t_E'] - baseline_energy(item['z'])
                    loss_e = ((p_E - target_E) / n_ats)**2
                    loss_f = torch.mean((p_F - item['t_F'])**2)
                    loss_s = torch.tensor(0.0, device=device)
                    if stress_target:
                        loss_s = torch.mean(
                            (stress_to_voigt(p_S) - stress_to_voigt(item['t_S']))**2
                        )
                    val_loss_accum += (
                        (config['energy_weight'] * loss_e)
                        + (force_weight * loss_f)
                        + (stress_weight * loss_s)
                    )

                val_metrics.update(
                    p_E, p_F, p_S, target_E, item['t_F'], item['t_S'],
                    stress_target, n_ats,
                )

        avg_val_loss = val_loss_accum / len(val_atoms)
        val_e, val_f, val_s, val_f_mse, val_f_mae, val_e_free, val_e_bias = val_metrics.get_metrics()
        avg_val_loss = float(avg_val_loss.detach().cpu())
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(avg_val_loss)
        history['train_energy_rmse'].append(float(tr_e))
        history['val_energy_rmse'].append(float(val_e))
        history['train_energy_rmse_offset_free'].append(float(tr_e_free))
        history['val_energy_rmse_offset_free'].append(float(val_e_free))
        history['train_energy_bias'].append(float(tr_e_bias))
        history['val_energy_bias'].append(float(val_e_bias))
        history['train_force_rmse'].append(float(tr_f))
        history['val_force_rmse'].append(float(val_f))
        history['train_stress_rmse'].append(float(tr_s))
        history['val_stress_rmse'].append(float(val_s))
        if scheduler_interval == 'epoch':
            scheduler.step()

        print(
            f"{epoch+1:5d} | "
            f"{avg_train_loss:10.4f} | {tr_e:10.2f} | {tr_e_free:9.2f} | {tr_f:12.6f} | {tr_f_mse:12.6f} | {tr_f_mae:12.6f} | {tr_s:10.4f} || "
            f"{avg_val_loss:10.4f} | {val_e:10.2f} | {val_e_free:10.2f} | {val_f:16.6f} | {val_f_mse:16.6f} | {val_s:12.6f}"
        )

        epochs_completed = epoch + 1
        if epoch + 1 >= early_stopping_min_epoch:
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_checkpoint(
                    config['model_save_path'],
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    energy_shift_per_atom,
                    atomic_energy_map,
                )
                best_checkpoint_saved = True
                print(f"New best validation loss at epoch {best_epoch}: {best_val_loss:.6f}")
            else:
                epochs_without_improvement += 1

        if ckpt_interval > 0 and (epoch + 1) % ckpt_interval == 0:
            ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                ckpt_path,
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                energy_shift_per_atom,
                atomic_energy_map,
            )

        if (
            early_stopping_patience > 0
            and epoch + 1 >= early_stopping_min_epoch
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}; best epoch was "
                f"{best_epoch} with validation loss {best_val_loss:.6f}."
            )
            break

    if not best_checkpoint_saved:
        save_checkpoint(
            config['model_save_path'],
            epochs_completed,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            energy_shift_per_atom,
            atomic_energy_map,
        )
        best_epoch = epochs_completed

    if config.get('save_last_checkpoint', True):
        root, extension = os.path.splitext(config['model_save_path'])
        last_path = config.get('last_model_save_path') or f"{root}_last{extension or '.pt'}"
        save_checkpoint(
            last_path,
            epochs_completed,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            energy_shift_per_atom,
            atomic_energy_map,
        )
    print(
        f"Training finished. Best checkpoint: {config['model_save_path']} "
        f"(epoch {best_epoch})."
    )

    if config.get('plot_training_curves', True):
        try:
            plot_metric_history(history, save_dir=config.get('plot_dir', 'plots'))
        except Exception as error:
            print(f"[PLOTTING] Could not create training curves: {error}")

if __name__ == "__main__": main()
