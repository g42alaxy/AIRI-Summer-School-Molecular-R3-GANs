import torch
from ase.db import connect
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
import typing as tp
from egnn.models import EGNN_dynamics_QM9
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
import numpy as np
from rdkit import Chem
from qm9.rdkit_functions import build_molecule
from rdkit.Contrib.SA_Score import sascorer
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# --- Params ---
class Args:
    r"""
    Project's config
    """
    def __init__(self):
        self.batch_size = 32
        self.include_charges = False
        self.augment_noise = 0
        self.data_augmentation = False
        self.ode_regularization = 0.0
        self.clip_grad = True
        self.conditioning = []
        self.n_report_steps = 10
        self.ema_decay = 0.999
        self.exp_name = "nablaDFT_EDM"
        self.break_train_epoch = False
        self.test_epochs = 5
        self.visualize_every_batch = 100
        self.num_workers = 0 
        self.save_epochs = 1
        self.lr = 1e-4
        self.wd = 0.0
        self.n_epochs = 100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.remove_h = False
        self.normalization_factor = 1
        self.dummy_int_dim = 0
        self.dummy_float_dim = 0
        self.in_node_nf = len(atom_types)
        self.context_node_nf = 0


# --- Dataset ---
class NablaDFTDataset(Dataset):
    """
    NablaDFT dataset wrapper
    :param db_path: path to DB file
    :param max_samples: max number of samples
    :param atom_types: list of atom types
    :param remove_h: whether to remove hydrogenes or not
    """

    def __init__(self,
                 db_path: str,
                 max_samples: int = 2000,
                 atom_types: tp.Optional[list[str]] = None,
                 remove_h: bool = False) -> None:
        self.db = connect(db_path)
        self.ids = [row.id for row in self.db.select()]
        self.max_samples = min(max_samples, len(self.ids)) if max_samples else len(self.ids)
        self.atom_types = atom_types
        self.remove_h = remove_h

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor] | Tensor]:
        real_id = self.ids[idx]
        row = self.db.get(real_id)

        positions = torch.tensor(row.positions, dtype=torch.float32)
        atomic_numbers = torch.tensor(row.numbers, dtype=torch.long)

        if self.remove_h:
            mask = (atomic_numbers != 1)
            positions = positions[mask]
            atomic_numbers = atomic_numbers[mask]

        one_hot_cat = torch.zeros(len(atomic_numbers), len(self.atom_types))
        for i, num in enumerate(atomic_numbers):
            if num in self.atom_types:
                one_hot_cat[i, self.atom_types.index(num)] = 1.
            else:
                raise ValueError(f"Atom number {num} not in predefined types")

        # categorical features only
        h_dict = {
            'categorical': one_hot_cat,
            'integer': torch.zeros(len(atomic_numbers), 0, dtype=torch.float),  # empty integer
            'float': torch.zeros(len(atomic_numbers), 0, dtype=torch.float)  # empty float
        }

        atom_mask = torch.ones(len(atomic_numbers), dtype=torch.float32)
        center_of_mass = positions.mean(dim=0, keepdim=True)
        positions_centered = positions - center_of_mass

        return {
            'positions': positions_centered,
            'h': h_dict,
            'atom_mask': atom_mask,
            'charges': torch.zeros(len(atomic_numbers))
        }


# --- DataLoaders ---
def collate_fn(batch) -> dict[str, dict[str, Tensor] | Tensor]:
    r"""
    Collate function for dataloaders
    :param batch: data batch
    :return: dict of parameters
    """
    max_n = max(item['positions'].shape[0] for item in batch)
    batch_size = len(batch)
    n_atom_types = batch[0]['h']['categorical'].shape[1]

    positions = torch.zeros((batch_size, max_n, 3))
    h_cat = torch.zeros((batch_size, max_n, n_atom_types))
    atom_mask = torch.zeros((batch_size, max_n))
    charges = torch.zeros((batch_size, max_n))

    for i, item in enumerate(batch):
        n_atoms = item['positions'].shape[0]
        positions[i, :n_atoms] = item['positions']
        h_cat[i, :n_atoms] = item['h']['categorical']
        atom_mask[i, :n_atoms] = item['atom_mask']
        charges[i, :n_atoms] = item['charges']

    edge_mask = (atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1)).bool()

    return {
        'positions': positions,
        'h': {
            'categorical': h_cat,
            'integer': torch.zeros(batch_size, max_n, 0),  # Пустой integer
            'float': torch.zeros(batch_size, max_n, 0)  # Пустой float
        },
        'atom_mask': atom_mask,
        'charges': charges,
        'edge_mask': edge_mask,
    }


# --- Train function ---
def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: optim.Optimizer) -> float:
    """
    Function for one train epoch
    :param model: diffusion model
    :param loader: data loader
    :param optimizer: optimizer
    :return: loss value
    """
    model.train()
    total_loss = 0

    for batch in loader:
        positions = batch['positions'].to(args.device)
        h_dict = {
            'categorical': batch['h']['categorical'].to(args.device),
            'integer': batch['h']['integer'].to(args.device),
            'float': batch['h']['float'].to(args.device)
        }
        node_mask = batch['atom_mask'].to(args.device).unsqueeze(-1)
        edge_mask = batch['edge_mask'].to(args.device)

        optimizer.zero_grad()

        loss_dict = model(
            x=positions,
            h=h_dict,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=None
        )

        if isinstance(loss_dict, dict):
            loss = loss_dict["loss"]
        elif isinstance(loss_dict, torch.Tensor):
            loss = loss_dict.mean()
        else:
            raise ValueError("Unknown loss type")
        loss.backward()

        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * positions.size(0)

    return total_loss / len(loader.dataset)


# --- Transformation of atom xyz to molecules
def onehot_to_type_indices(onehot_rows: torch.Tensor) -> list[np.ndarray]:
    r"""
    One-hot to atom types
    :param onehot_rows: one-hot encoded atoms
    :return:
    """
    if hasattr(onehot_rows, 'cpu'):
        onehot_rows = onehot_rows.cpu().numpy()
    return [np.int64(np.argmax(row)) for row in onehot_rows]


def to_rdkit_mol(z_onehot_rows: torch.Tensor, pos: torch.Tensor) -> Chem.Mol:
    r"""
    Transform XYZ to molecules
    :param z_onehot_rows:
    :param pos:
    :return:
    """
    dataset_info = {
        'name': 'geom',
        "atom_decoder": ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br']
    }
    type_indices = onehot_to_type_indices(z_onehot_rows)
    return build_molecule(pos, type_indices, dataset_info=dataset_info)

def is_valid(mol: Chem.Mol) -> bool:
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


def calculate_quality_score(smiles: tp.Iterable[str]) -> float:
    """
    :param smiles: SMILES strings
    :return: Fraction of unique and valid molecules with qed >= 0.6 and sascore <= 4.
    """
    valid_mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)

    if not valid_mols:
        return 0.0
    passed = 0
    for mol in set(valid_mols):
        qed_score = QED.qed(mol)
        sa_score = sascorer.calculateScore(mol)
        if qed_score >= 0.6 and sa_score <= 4:
            passed += 1

    return passed / len(smiles)


def compute_metrics(mols: list[Chem.Mol]) -> dict[str, float]:
    """
    Compute metrics for a list of molecules
    :param mols: molecules
    :return: dictionary of metrics
    """
    metrics = []
    for mol in mols:
        if mol is not None:
            metrics.append({
                'qed': QED.qed(mol),
                'logp': Descriptors.MolLogP(mol),
                'sa': sascorer.calculateScore(mol)
            })
        else:
            metrics.append(None)
    return metrics

if __name__ == '__main__':
    # --- Main configuration ---
    args = Args()

    n_samples = 100
    num_atoms = 20
    device = args.device

    n_nodes = num_atoms

    # node_mask: [batch_size, num_atoms]
    node_mask = torch.ones((n_samples, num_atoms), dtype=torch.float).to(device)
    node_mask_for_sample = node_mask.unsqueeze(-1)

    # edge_mask: [batch_size, num_atoms, num_atoms]
    edge_mask = torch.ones((n_samples, num_atoms, num_atoms), dtype=torch.bool).to(device)

    context = None

    # --- Defining the atom types ---
    db_path = '../train_100k_v2_formation_energy_w_forces.db'
    full_db = connect(db_path)
    all_atom_types = set()
    for row in full_db.select():
        nums = list(row.numbers)
        if args.remove_h:
            nums = [n for n in nums if n != 1]
        all_atom_types.update(nums)
    atom_types = sorted(all_atom_types)
    print(f"Found {len(atom_types)} atom types: {atom_types}")

    # --- Datasets and dataloaders ---
    dataset_train_full = NablaDFTDataset(
        db_path=db_path,
        max_samples=len(full_db),
        atom_types=atom_types,
        remove_h=args.remove_h
    )

    train_loader = DataLoader(
        dataset_train_full,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    print(f"Train size: {len(dataset_train_full)}")

    # --- Model ---

    # Normalization parameters
    norm_values = (1, 4, 1)
    norm_biases = (0, 0, 0)

    dynamics_model = EGNN_dynamics_QM9(
        in_node_nf=args.in_node_nf,
        context_node_nf=args.context_node_nf,
        n_dims=3,
        hidden_nf=128,
        device=args.device,
        act_fn=torch.nn.SiLU(),
        attention=True,
        tanh=True,
        norm_constant=1,
        inv_sublayers=2,
        sin_embedding=False
    ).to(args.device)

    diffusion_model = EnVariationalDiffusion(
        dynamics=dynamics_model,
        in_node_nf=args.in_node_nf,
        n_dims=3,
        timesteps=1000,
        norm_values=norm_values,
        norm_biases=norm_biases,
        include_charges=False
    ).to(args.device)

    optimizer = optim.Adam(diffusion_model.parameters(), lr=args.lr)


    # --- Create directories for model checkpoints ---
    os.makedirs("saved_models", exist_ok=True)

    # --- Training process ---
    losses = []
    uniquenesses = []
    novelties = []
    validities = []

    data = pd.read_csv('../summary.csv.gz', compression='gzip')
    train_smiles_set = data.SMILES.unique()

    qeds, logs, sas, quals = [], [], [], []

    for epoch in range(args.n_epochs):
        t_start = time.time()
        diffusion_model.train()
        train_loss = train_epoch(diffusion_model, train_loader, optimizer)
        print(f"Epoch {epoch+1}/{args.n_epochs} | Train Loss: {train_loss:.4f} | Time: {time.time()-t_start:.1f}s")
        losses.append(train_loss)
        diffusion_model.eval()
        with torch.no_grad():
            generated_data = diffusion_model.sample(
                n_samples=n_samples,
                n_nodes=n_nodes,
                node_mask=node_mask_for_sample,
                edge_mask=edge_mask,
                context=context,
                fix_noise=False
            )

            mol_list = [to_rdkit_mol(z, pos) for z, pos in zip(generated_data[1]['categorical'], generated_data[0])]
            valid_flags = [is_valid(m) for m in mol_list]
            valid_mols = [m for m,v in zip(mol_list, valid_flags) if v]
            metrics_list = compute_metrics(valid_mols)

            # Uniqueness & Novelty
            smiles_list = [Chem.MolToSmiles(m) for m in valid_mols]
            unique_smiles_set = set(smiles_list)
            quality = calculate_quality_score(smiles_list)
            uniqueness = len(unique_smiles_set)/max(1,len(generated_data[0]))
            novelty = len([s for s in unique_smiles_set if s not in train_smiles_set])/max(1,len(unique_smiles_set))

            mean_qed = np.mean([m['qed'] for m in metrics_list if m is not None])
            mean_logp = np.mean([m['logp'] for m in metrics_list if m is not None])
            mean_sa = np.mean([m['sa'] for m in metrics_list if m is not None])

            validities.append(sum(np.array(valid_flags, dtype=int)) / len(generated_data[0]))
            uniquenesses.append(uniqueness)
            novelties.append(novelty)
            plt.plot(validities, label='Validity')
            plt.plot(uniquenesses, label='Uniqueness')
            plt.plot(novelties, label='Novelty')
            plt.legend(loc="upper left")
            plt.savefig('metrics.png', dpi=150)
            plt.clf()

            qeds.append(mean_qed)
            logs.append(mean_logp)
            sas.append(mean_sa)
            quals.append(quality)
            fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(12, 2.5))
            fig.tight_layout()
            axs[0].plot(qeds)
            axs[0].set_title('QED')
            axs[1].plot(logs)
            axs[1].set_title('LogP')
            axs[2].plot(sas)
            axs[2].set_title('SA score')
            axs[3].plot(quals)
            axs[3].set_title('Quality')
            plt.savefig('chem_metrics.png', dpi=150)
            plt.clf()

        if (epoch+1) % args.save_epochs == 0:
            save_path = f"saved_models/edm_epoch_{epoch+1}_large.pt"
            torch.save(diffusion_model.state_dict(), save_path)
            print(f"Model saved to {save_path}")