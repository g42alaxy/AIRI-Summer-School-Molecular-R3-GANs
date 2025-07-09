from rdkit import Chem
from rdkit.Chem import QED
import pandas as pd
from quality.sascorer import sascorer

import rdkit
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def calculate_quality_score(smiles_column: pd.Series) -> float:
    """
    Estimates the number of unique valid mols with qed >= 0.6 and sascore <= 4.
    
    Args:
        smiles_column (pd.Series): smiles строки

    Returns:
        float: quality score 
    """
    valid_mols = []
    for smi in smiles_column.dropna():
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