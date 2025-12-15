import yaml
import torch
import torch.optim as optim
import numpy as np
import time
from ase.io import read
from flashace.model import FlashACE
from flashace.plotting import plot_training_results
from flashace.utils import frozen_parameter_grads
from ase.neighborlist import neighbor_list
from torch.utils.data import DataLoader, Dataset, random_split

# --- STABILITY SETTINGS ---
# Disable TF32 to prevent potential TensorCore precision crashes in e3nn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class AtomisticDataset(Dataset):
    def __init__(self, atoms_list, r_max):
        self.atoms_list = atoms_list
        self.r_max = r_max
        
    def __len__(self): return len(self.atoms_list)
    
    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        
        # Geometry
        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        vol = torch.tensor(atoms.get_volume(), dtype=torch.float32)
        
        # Targets
        t_E = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        t_F = torch.tensor(atoms.get_forces(), dtype=torch.float32)
        
        # Stress (Robust Load)
        s_obj = None
        if atoms.calc is not None and 'stress' in atoms.calc.results:
            s_obj = atoms.calc.results['stress']
        elif 'stress' in atoms.info:
            s_obj = atoms.info['stress']
        elif 'virial' in atoms.info:
            s_obj = atoms.info['virial'] / atoms.get_volume()
            
        if s_obj is None:
            s_voigt = np.zeros((3,3))
        else:
            s_voigt = np.array(s_obj)

        if s_voigt.shape == (6,):
            s_mat = np.array([[s_voigt[0], s_voigt[5], s_voigt[4]],
                              [s_voigt[5], s_voigt[1], s_voigt[3]],
                              [s_voigt[4], s_voigt[3], s_voigt[2]]])
            t_S = torch.tensor(s_mat, dtype=torch.float32)
        else:
            t_S = torch.tensor(s_voigt, dtype=torch.float32)
            if len(t_S.shape) == 1: t_S = t_S.view(3,3)
        
        # Neighbors
        i, j = neighbor_list('ij', atoms, self.r_max)
        edge_index = torch.stack([torch.tensor(i), torch.tensor(j)], dim=0)
        
        return {'z':z, 'pos':pos, 'edge_index':edge_index, 'volume':vol, 't_E':t_E, 't_F':t_F, 't_S':t_S}
    
    @staticmethod
    def collate_fn(batch): return batch

class MetricTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.sse_e = 0.0; self.sse_f = 0.0; self.sse_s = 0.0
        self.n_atoms = 0; self.n_force_comp = 0; self.n_stress_comp = 0
    def update(self, p_E, p_F, p_S, t_E, t_F, t_S, n_ats):
        err_e = (p_E - t_E).item() / n_ats
        self.sse_e += err_e**2 * n_ats
        self.sse_f += (p_F - t_F).pow(2).sum().item()
        if torch.norm(t_S) > 1e-6:
             self.sse_s += (p_S - t_S).pow(2).sum().item()
             self.n_stress_comp += 9
        self.n_atoms += n_ats
        self.n_force_comp += n_ats * 3
    def get_metrics(self):
        rmse_e = np.sqrt(self.sse_e / self.n_atoms) if self.n_atoms > 0 else 0.0
        rmse_f = np.sqrt(self.sse_f / self.n_force_comp) if self.n_force_comp > 0 else 0.0
        rmse_s = np.sqrt(self.sse_s / self.n_stress_comp) if self.n_stress_comp > 0 else 0.0
        return rmse_e * 1000, rmse_f, rmse_s

def compute_mean_energy_per_atom(atoms_seq):
    total_energy = 0.0
    total_atoms = 0
    for atoms in atoms_seq:
        total_energy += atoms.get_potential_energy()
        total_atoms += len(atoms)
    return (total_energy / total_atoms) if total_atoms > 0 else 0.0

def main():
    print("--- Loading config.yaml ---")
    with open("config.yaml", "r") as f: config = yaml.safe_load(f)
    device = config['device']
    
    print(f"Reading data from {config['train_file']}...")
    all_atoms = read(config['train_file'], index=":")
    
    if config['valid_file']:
        val_atoms = read(config['valid_file'], index=":")
        train_atoms = all_atoms
    else:
        val_len = max(1, int(len(all_atoms) * config.get('val_split', 0.1)))
        train_len = len(all_atoms) - val_len
        train_atoms, val_atoms = random_split(
            all_atoms, [train_len, val_len], 
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Random Split: {train_len} Training | {val_len} Validation")

    if config.get('energy_shift_per_atom') is not None:
        energy_shift_per_atom = float(config['energy_shift_per_atom'])
        print(f"Using user-provided energy shift per atom: {energy_shift_per_atom:.6f} eV")
    else:
        energy_shift_per_atom = compute_mean_energy_per_atom(train_atoms)
        print(f"Computed mean energy per atom for normalization: {energy_shift_per_atom:.6f} eV")

    energy_shift = torch.tensor(energy_shift_per_atom, dtype=torch.float32, device=device)

    # DATALOADERS
    train_ds = AtomisticDataset(train_atoms, config['r_max'])
    val_ds = AtomisticDataset(val_atoms, config['r_max'])

    # Reduced workers to prevent CPU overhead issues
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, shuffle=True,
                              num_workers=2, pin_memory=True)
    
    valid_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, num_workers=2)

    print("--- Initializing FlashACE ---")
    model = FlashACE(
        r_max=config['r_max'], l_max=config['l_max'], num_radial=config['num_radial'], 
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15, verbose=True)

    history = {'train_loss':[], 'val_loss':[]}
    
    print(f"{'Epoch':>5} | {'Loss':>8} | {'E (meV)':>8} | {'F (eV/A)':>8} | {'S (eV/A³)':>8} || {'Val Loss':>8} | {'Val E':>8} | {'Val F':>8}")
    print("-" * 95)
    
    for epoch in range(config['epochs']):
        model.train()
        train_metrics = MetricTracker()
        total_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            
            # --- GRADIENT ACCUMULATION (FP32) ---
            for item in batch:
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        item[k] = v.to(device, non_blocking=True)

                # Standard Forward (No AMP)
                p_E, p_F, p_S = model(item, training=True)
                n_ats = len(item['z'])

                target_E = item['t_E'] - energy_shift * n_ats
                loss_e = ((p_E - target_E) / n_ats)**2
                loss_f = torch.mean((p_F - item['t_F'])**2)
                loss_s = torch.tensor(0.0, device=device)
                if torch.norm(item['t_S']) > 1e-6:
                    loss_s = torch.mean((p_S - item['t_S'])**2)
                
                loss_item = (config['energy_weight']*loss_e) + \
                            (config['forces_weight']*loss_f) + \
                            (config['stress_weight']*loss_s)
                
                # Normalize and Backward
                loss_batch = loss_item / len(batch)
                loss_batch.backward()

                batch_loss += loss_item.item()

                with torch.no_grad():
                    pred_E_abs = p_E + energy_shift * n_ats
                    train_metrics.update(pred_E_abs, p_F, p_S, item['t_E'], item['t_F'], item['t_S'], n_ats)

            optimizer.step()
            total_loss += batch_loss

        avg_train_loss = total_loss / len(train_atoms)
        tr_e, tr_f, tr_s = train_metrics.get_metrics()
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()

        # Keep gradients disabled for parameters while still allowing
        # autograd to track atomic positions for forces/stresses.
        with frozen_parameter_grads(model):
            val_metrics = MetricTracker()
            val_loss_accum = 0.0

            for batch in valid_loader:
                for item in batch:
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor):
                            item[k] = v.to(device, non_blocking=True)

                    # Compute stress only when requested to avoid extra autograd work
                    wants_stress = config['stress_weight'] > 0 and torch.norm(item['t_S']) > 1e-6
                    p_E, p_F, p_S = model(item, training=wants_stress)
                    n_ats = len(item['z'])
                    target_E = item['t_E'] - energy_shift * n_ats
                    loss_e = ((p_E - target_E) / n_ats)**2
                    loss_f = torch.mean((p_F - item['t_F'])**2)
                    loss_s = torch.tensor(0.0, device=device)
                    if wants_stress:
                        loss_s = torch.mean((p_S - item['t_S'])**2)

                    val_loss = (
                        (config['energy_weight'] * loss_e)
                        + (config['forces_weight'] * loss_f)
                        + (config['stress_weight'] * loss_s)
                    )
                    val_loss_accum += float(val_loss.item())

                    with torch.no_grad():
                        pred_E_abs = p_E + energy_shift * n_ats
                        val_metrics.update(pred_E_abs, p_F, p_S, item['t_E'], item['t_F'], item['t_S'], n_ats)

            avg_val_loss = val_loss_accum / len(val_atoms)
            val_e, val_f, val_s = val_metrics.get_metrics()
            history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"{epoch+1:5d} | {avg_train_loss:8.4f} | {tr_e:8.2f} | {tr_f:8.4f} | {tr_s:8.4f} || {avg_val_loss:8.4f} | {val_e:8.2f} | {val_f:8.4f}")

    # SAVE
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'r_max': config['r_max'],
            'l_max': config['l_max'],
            'num_radial': config['num_radial'],
            'hidden_dim': config['hidden_dim'],
            'num_layers': config['num_layers'],
            'energy_shift_per_atom': energy_shift_per_atom
        }
    }
    torch.save(checkpoint, config['model_save_path'])
    print(f"Training Finished. Saved to {config['model_save_path']}")

if __name__ == "__main__": main()
