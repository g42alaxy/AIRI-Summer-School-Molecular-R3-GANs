import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
import sascorer
from tqdm import tqdm

# --- Параметры ---
class Args:
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
        self.exp_name = "QM9_EDM"
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
        self.in_node_nf = 5  # H, C, N, O, F
        self.context_node_nf = 0

args = Args()

# --- Атомные типы для QM9 ---
atom_types = [1, 6, 7, 8, 9]  # H, C, N, O, F
atomic_num_to_idx = {num: idx for idx, num in enumerate(atom_types)}
atom_decoder = ['H', 'C', 'N', 'O', 'F']

# --- Загрузка SMILES и генерация 3D структур ---
def generate_conformers(smiles_file):
    smiles_list = []
    with open(smiles_file, 'r') as f:
        for line in f:
            smiles = line.strip()
            if smiles:
                smiles_list.append(smiles)
    
    mols = []
    print(f"Generating 3D conformers for {len(smiles_list)} molecules...")
    for smi in tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
                
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            
            if mol.GetNumConformers() == 0:
                continue
                
            mols.append(mol)
        except Exception as e:
            print(f"Error processing {smi}: {str(e)}")
    
    print(f"Successfully generated {len(mols)} 3D structures")
    return mols

# --- Датасет QM9 ---
class QM9Dataset(Dataset):
    def __init__(self, mols, atomic_num_to_idx):
        self.mols = mols
        self.atomic_num_to_idx = atomic_num_to_idx
        
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, idx):
        mol = self.mols[idx]
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        positions = torch.tensor(positions, dtype=torch.float32)
        
        # Центрирование координат относительно центра масс
        center_of_mass = positions.mean(dim=0, keepdim=True)
        positions_centered = positions - center_of_mass
        
        atomic_numbers = []
        for atom in mol.GetAtoms():
            atomic_numbers.append(atom.GetAtomicNum())
        
        one_hot = torch.zeros(len(atomic_numbers), len(self.atomic_num_to_idx))
        for i, z in enumerate(atomic_numbers):
            if z in self.atomic_num_to_idx:
                one_hot[i, self.atomic_num_to_idx[z]] = 1.0
        
        atom_mask = torch.ones(len(atomic_numbers), dtype=torch.float32)
        
        return {
            'positions': positions_centered,  # Используем центрированные координаты
            'h': {
                'categorical': one_hot,
                'integer': torch.zeros(len(atomic_numbers), 0, dtype=torch.float),
                'float': torch.zeros(len(atomic_numbers), 0, dtype=torch.float)
            },
            'atom_mask': atom_mask,
            'charges': torch.zeros(len(atomic_numbers))
        }

# --- Collate функция ---
def collate_fn(batch):
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
            'integer': torch.zeros(batch_size, max_n, 0),
            'float': torch.zeros(batch_size, max_n, 0)
        },
        'atom_mask': atom_mask,
        'charges': charges,
        'edge_mask': edge_mask,
    }

# --- Генерация данных ---
mols = generate_conformers('qm9_5k.smi')
dataset = QM9Dataset(mols, atomic_num_to_idx)

# Разделение на train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=collate_fn
)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# --- Модель ---
from egnn.models import EGNN_dynamics_QM9
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion

# Параметры нормализации для QM9
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

# --- Функция обучения ---
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.
    
    for batch in tqdm(loader, desc="Training"):
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

# --- Функции для оценки ---
def onehot_to_type_indices(onehot_rows):
    if hasattr(onehot_rows, 'cpu'):
        onehot_rows = onehot_rows.cpu().numpy()
    return [np.argmax(row) for row in onehot_rows]

def build_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info['atom_decoder']
    mol = Chem.RWMol()
    
    for atom_type in atom_types:
        a = Chem.Atom(atom_decoder[atom_type])
        mol.AddAtom(a)
    
    conf = Chem.Conformer(len(positions))
    for i in range(len(positions)):
        conf.SetAtomPosition(i, positions[i].tolist())
    mol.AddConformer(conf)
    
    return mol

def to_rdkit_mol(z_onehot_rows, pos):
    dataset_info = {'atom_decoder': atom_decoder}
    type_indices = onehot_to_type_indices(z_onehot_rows)
    return build_molecule(pos, type_indices, dataset_info)

def is_valid(mol):
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False

def calculate_quality_score(smiles_column: pd.Series) -> float:
    """
    Estimates the number of unique valid mols with qed >= 0.6 and sascore <= 4.
    
    Args:
        smiles_column (pd.Series): smiles строки

    Returns:
        float: quality score 
    """
    valid_mols = []
    for smi in smiles_column:
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)

    unique_mols = set(valid_mols)

    passed = 0
    for mol in unique_mols:
        try:
            qed_score = QED.qed(mol)
            sa_score = sascorer.calculateScore(mol)
            if qed_score >= 0.6 and sa_score <= 4:
                passed += 1
        except:
            pass

    return passed / len(valid_mols)

def compute_metrics(mols):
    metrics = []
    for mol in mols:
        if mol is not None:
            try:
                metrics.append({
                    'qed': QED.qed(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'sa': sascorer.calculateScore(mol) 
                })
            except:
                metrics.append(None)
        else:
            metrics.append(None)
    return metrics

# --- Подготовка к обучению ---
os.makedirs("saved_models", exist_ok=True)

# Сбор SMILES тренировочного набора для оценки новизны
train_smiles_set = set()
for mol in train_dataset:
    try:
        rdmol = to_rdkit_mol(
            mol['h']['categorical'],
            mol['positions'].numpy()
        )
        if is_valid(rdmol):
            smi = Chem.MolToSmiles(rdmol, isomericSmiles=False)
            train_smiles_set.add(smi)
    except:
        continue

# Гистограмма размеров молекул для сэмплирования
train_lengths = [len(item['positions']) for item in train_dataset]

# --- Обучение ---
losses = []
uniquenesses = []
novelties = []
validities = []
qeds, logs, sas, quals = [], [], [], []

for epoch in range(args.n_epochs):
    t_start = time.time()
    
    # Обучение
    train_loss = train_epoch(diffusion_model, train_loader, optimizer)
    losses.append(train_loss)
    
    # Валидация и генерация
    if (epoch + 1) % args.test_epochs == 0:
        diffusion_model.eval()
        n_samples = 100
        
        # Сэмплирование размеров молекул
        n_nodes_list = np.random.choice(train_lengths, size=n_samples, replace=True)
        max_n_nodes = max(n_nodes_list)
        
        node_mask = torch.zeros((n_samples, max_n_nodes))
        for i, n in enumerate(n_nodes_list):
            node_mask[i, :n] = 1
        node_mask = node_mask.to(args.device).unsqueeze(-1)
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        
        with torch.no_grad():
            generated_data = diffusion_model.sample(
                n_samples=n_samples,
                n_nodes=max_n_nodes,  # Исправлено: передаем скаляр (максимальное число атомов)
                node_mask=node_mask,
                edge_mask=edge_mask,
                context=None,
                fix_noise=False
            )
        
        # Оценка качества
        mol_list = []
        for z, pos in zip(generated_data[1]['categorical'], generated_data[0]):
            try:
                mol = to_rdkit_mol(z, pos.cpu().numpy())
                mol_list.append(mol)
            except:
                mol_list.append(None)
        
        valid_flags = [is_valid(m) if m is not None else False for m in mol_list]
        valid_mols = [m for m, v in zip(mol_list, valid_flags) if v]
        
        # Вычисление метрик
        validity = np.mean(valid_flags)
        validities.append(validity)
        
        smiles_list = []
        for mol in valid_mols:
            try:
                smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                smiles_list.append(smi)
            except:
                continue
        
        unique_smiles = set(smiles_list)
        uniqueness = len(unique_smiles) / max(1, len(mol_list))
        novelty = len([s for s in unique_smiles if s not in train_smiles_set]) / max(1, len(unique_smiles))
        
        uniquenesses.append(uniqueness)
        novelties.append(novelty)
        
        quality = calculate_quality_score(smiles_list) if smiles_list else 0.0
        quals.append(quality)
        
        metrics_list = compute_metrics(valid_mols)
        if metrics_list:
            mean_qed = np.mean([m['qed'] for m in metrics_list if m is not None])
            mean_logp = np.mean([m['logp'] for m in metrics_list if m is not None])
            mean_sa = np.mean([m['sa'] for m in metrics_list if m is not None])
        else:
            mean_qed, mean_logp, mean_sa = 0.0, 0.0, 0.0
        
        qeds.append(mean_qed)
        logs.append(mean_logp)
        sas.append(mean_sa)
        
        # Визуализация метрик
        plt.plot(validities, label='Validity')
        plt.plot(uniquenesses, label='Uniqueness')
        plt.plot(novelties, label='Novelty')
        plt.legend(loc="upper left")
        plt.savefig('metrics_qm9.png')
        plt.clf()

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
        plt.savefig('chem_metrics_qm9.png', dpi=150)
        plt.clf()
    
    # Логирование и сохранение
    epoch_time = time.time() - t_start
    print(f"Epoch {epoch+1}/{args.n_epochs} | Loss: {train_loss:.4f} | "
          f"Validity: {validities[-1] if validities else 0:.2f} | "
          f"Time: {epoch_time:.1f}s")
    
    if (epoch + 1) % args.save_epochs == 0:
        save_path = f"saved_models/edm_epoch_{epoch+1}.pt"
        torch.save(diffusion_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

print("Training completed!")