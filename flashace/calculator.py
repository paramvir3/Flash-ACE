import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from .model import FlashACE
from .utils import frozen_parameter_grads

class FlashACECalculator(Calculator):
    """
    ASE Calculator for FlashACE.
    Uses standard ASE neighbor lists (highly compatible).
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path="model.pt", device=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        # 1. Device Setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading FlashACE from {model_path} on {self.device}...")

        # 2. Load Model & Config
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if 'config' not in checkpoint:
            raise KeyError("Model file missing 'config'. Please retrain with updated train.py.")

        conf = checkpoint['config']

        self.energy_shift_per_atom = float(conf.get('energy_shift_per_atom', 0.0))

        # Ensure cutoff is float
        self.r_max = float(conf['r_max'])

        # 3. Initialize Architecture
        self.model = FlashACE(
            r_max=self.r_max,
            l_max=conf['l_max'],
            num_radial=conf['num_radial'],
            hidden_dim=conf['hidden_dim'],
            num_layers=conf['num_layers']
        )
        
        # 4. Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        # Standard ASE setup
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 1. Neighbor List (Standard ASE)
        i, j = neighbor_list('ij', atoms, self.r_max)
        edge_index = torch.stack([torch.tensor(i), torch.tensor(j)], dim=0).to(self.device)
        
        # 2. Prepare Data
        z = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device)
        pos = torch.tensor(atoms.positions, dtype=torch.float32, device=self.device)
        
        # Volume handling (Use 1.0 for non-periodic systems)
        if atoms.pbc.any():
            vol = atoms.get_volume()
        else:
            vol = 1.0

        data = {
            'z': z, 'pos': pos, 'edge_index': edge_index,
            'volume': torch.tensor(vol, dtype=torch.float32, device=self.device)
        }
        
        # 3. Run Model
        calc_stress = 'stress' in properties

        # If calculating stress, enable gradients w.r.t cell (training=True)
        with frozen_parameter_grads(self.model):
            if calc_stress:
                pred_E, pred_F, pred_S = self.model(data, training=True)
            else:
                pred_E, pred_F, _ = self.model(data, training=False)

        energy_shift = self.energy_shift_per_atom * len(atoms)
        pred_E = pred_E + energy_shift

        # 4. Store Results
        self.results['energy'] = pred_E.item()
        self.results['forces'] = pred_F.detach().cpu().numpy()
        
        if calc_stress:
            S_mat = pred_S.detach().cpu().numpy()
            # Convert 3x3 to Voigt (xx, yy, zz, yz, xz, xy)
            self.results['stress'] = np.array([
                S_mat[0,0], S_mat[1,1], S_mat[2,2],
                S_mat[1,2], S_mat[0,2], S_mat[0,1]
            ])
