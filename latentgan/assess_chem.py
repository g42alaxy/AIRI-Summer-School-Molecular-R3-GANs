import pandas as pd
import numpy as np

from quality.quality import calculate_quality_score
from quality.sascorer import sascorer

import rdkit
from rdkit import Chem
from rdkit.Chem import QED, Crippen

import sys

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def synthetic_accessibility_score_scores(mol):
    sas_ = sascorer.calculateScore(mol)
    if sas_ > 10:
        sas_ = 10

    return np.clip(remap(sas_, 5, 1.5), 0.0, 1.0)

def water_octanol_partition_coefficient_scores(mol):
    log_p = Crippen.MolLogP(mol)
    if log_p is None:
        log_p = 3
    
    return np.clip(remap(log_p, -2.12178879609, 6.0429063424), 0.0, 1.0)

if __name__ == "__main__": 
    file = sys.argv[1]
    
    with open(file, 'r') as f:
        data = f.readlines()
    
    data = pd.Series(data[1:])
    qeds   = []
    sases  = []
    log_ps = []
    mols   = []
    quality_counter = 0
    
    for i in range(len(data)):
        try:
            mol = Chem.MolFromSmiles(data[i])
            
            sas_value = sascorer.calculateScore(mol)
            qed_value = QED.qed(mol)

            if mol not in mols:
                mols.append(mol)
                if qed_value >= 0.6 and sas_value <= 4:
                    quality_counter +=1 
            
            qeds.append(qed_value)
            sases.append(synthetic_accessibility_score_scores(mol))
            log_ps.append(water_octanol_partition_coefficient_scores(mol))
        except:
            qeds.append(0)
            sases.append(0)
            log_ps.append(0)
    
    
    results = {
        'mean QED'      : np.mean(qeds),
        'mean SA'       : np.mean(sases),
        'mean logP'     : np.mean(log_ps),
        'quality score' : quality_counter / len(data)
    }

    print(results)