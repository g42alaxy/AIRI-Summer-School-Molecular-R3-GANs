import gzip
import math
import pickle

import numpy as np
import pandas as pd
import rdkit

from pysmiles import read_smiles
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, Crippen, Draw
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
from util_dir.utils_io import random_string
from collections import Counter

NP_model = pickle.load(gzip.open("data/NP_score.pkl.gz"))
SA_model = {
    i[j]: float(i[0])
    for i in pickle.load(gzip.open("data/SA_score.pkl.gz"))
    for j in range(1, len(i))
}


def get_mol(smiles_or_mol):
    """
    Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol
    
def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def mapper(n_jobs):
    """
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    """
    if n_jobs == 1:

        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map



class MolecularMetrics(object):
    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ""

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ""
        return x is not None and "*" not in s and "." not in s and s != ""
        
    @staticmethod 
    def valid_scores(mols, n_jobs=1):
        rd_mols = mapper(n_jobs)(get_mol, mols)
        return np.array([mol is not None for mol in rd_mols], dtype=float)
    
    @staticmethod
    def valid_total_score(mols, n_jobs=1):
        return MolecularMetrics.valid_scores(mols, n_jobs).mean()

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(
                map(
                    lambda x: MolecularMetrics.valid_lambda(x)
                    and Chem.MolToSmiles(x) not in data.smiles,
                    mols,
                )
            )
        )

    @staticmethod
    def novel_filter(mols, data):
        return list(
            filter(
                lambda x: MolecularMetrics.valid_lambda(x)
                and Chem.MolToSmiles(x) not in data.smiles,
                mols,
            )
        )

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(
            MolecularMetrics.valid_filter(mols), data
        ).mean()

    @staticmethod
    def unique_scores(mols, n_jobs=1):
        canonic = mapper(n_jobs)(canonic_smiles, mols)
        counts = Counter([s for s in canonic if s is not None])
        return np.array(
            [1.0 if s is not None and counts[s] == 1 else 0.0 for s in canonic],
            dtype=np.float32,
        )
    
    @staticmethod
    def unique_total_score(mols, n_jobs=1, check_validity=False):
        """
        Computes a number of unique molecules
        Parameters:
            gen: list of SMILES
            k: compute unique@k
            n_jobs: number of threads for calculation
            check_validity: raises ValueError if invalid molecules are present
        """
        canonic = set(mapper(n_jobs)(canonic_smiles, mols))
        if None in canonic and check_validity:
            raise ValueError("Invalid molecule passed to unique@k")
        return len(canonic) / len(mols)

    # @staticmethod
    # def novel_and_unique_total_score(mols, data):
    #     return ((MolecularMetrics.unique_scores(mols) == 1).astype(float) * MolecularMetrics.novel_scores(mols,
    #                                                                                                       data)).sum()
    #
    # @staticmethod
    # def reconstruction_scores(data, model, session, sample=False):
    #
    #     m0, _, _, a, x, _, f, _, _ = data.next_validation_batch()
    #     feed_dict = {model.edges_labels: a, model.nodes_labels: x, model.node_features: f, model.training: False}
    #
    #     try:
    #         feed_dict.update({model.variational: False})
    #     except AttributeError:
    #         pass
    #
    #     n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
    #         model.nodes_argmax, model.edges_argmax], feed_dict=feed_dict)
    #
    #     n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    #
    #     m1 = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
    #
    #     return np.mean([float(Chem.MolToSmiles(m0_) == Chem.MolToSmiles(m1_)) if m1_ is not None else 0
    #             for m0_, m1_ in zip(m0, m1)])

    @staticmethod
    def natural_product_scores(mols, norm=False):
        # calculating the score
        scores = [
            sum(
                NP_model.get(bit, 0)
                for bit in Chem.rdMolDescriptors.GetMorganFingerprint(
                    mol, 2
                ).GetNonzeroElements()
            )
            / float(mol.GetNumAtoms())
            if mol is not None
            else None
            for mol in mols
        ]

        # preventing score explosion for exotic molecules
        scores = list(
            map(
                lambda score: score
                if score is None
                else (
                    4 + math.log10(score - 4 + 1)
                    if score > 4
                    else (-4 - math.log10(-4 - score + 1) if score < -4 else score)
                ),
                scores,
            )
        )

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = (
            np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores
        )

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, n_jobs=1, norm=False):
        rd_mols = mapper(n_jobs)(get_mol, mols)

        qeds = []
        for mol in rd_mols:
            try:
                if mol is not None:
                    qeds.append(QED.qed(mol))
                else: 
                    qeds.append(0.0) 
            except:
                qeds.append(0.0)
            
        return np.array(qeds)

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [
            MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol))
            if mol is not None
            else None
            for mol in mols
        ]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = (
            np.clip(
                MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0
            )
            if norm
            else scores
        )

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms**1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.0

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = (
            0.0
            - sizePenalty
            - stereoPenalty
            - spiroPenalty
            - bridgePenalty
            - macrocyclePenalty
        )

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.0
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * 0.5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
        # smooth the 10-end
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        if sascore > 10.0:
            sascore = 10.0
        elif sascore < 1.0:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [
            MolecularMetrics._compute_SAS(mol) if mol is not None else None
            for mol in mols
        ]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = (
            np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0)
            if norm
            else scores
        )

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        fps = [
            Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            for mol in rand_mols
        ]

        scores = np.array(
            list(
                map(
                    lambda x: MolecularMetrics.__compute_diversity(x, fps)
                    if x is not None
                    else 0,
                    mols,
                )
            )
        )
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 4, nBits=2048
        )
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):
        scores = (
            MolecularMetrics.constant_bump(
                MolecularMetrics.water_octanol_partition_coefficient_scores(
                    mols, norm=True
                ),
                0.210,
                0.945,
            )
            + MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            + MolecularMetrics.novel_scores(mols, data)
            + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3
        ) / 4

        return scores

    @staticmethod
    def quality_scores(mols):
        valid_scores  = MolecularMetrics.valid_scores(mols).astype(bool)
        unique_scores = MolecularMetrics.unique_scores(mols).astype(bool)
        qeds_scores   = MolecularMetrics.quantitative_estimation_druglikeness_scores(mols)
        sas_scores    = np.array(MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=False))
    
        return ((qeds_scores >= 0.6).astype(bool) & (sas_scores <= 4).astype(bool) & (valid_scores) & (unique_scores)).mean()

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(
            condlist=[x <= x_low, x >= x_high],
            choicelist=[
                np.exp(-((x - x_low) ** 2) / decay),
                np.exp(-((x - x_high) ** 2) / decay),
            ],
            default=np.ones_like(x),
        )

def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


def classification_report(data, model, session, sample=False):
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    n, e = session.run(
        [model.nodes_gumbel_argmax, model.edges_gumbel_argmax]
        if sample
        else [model.nodes_argmax, model.edges_argmax],
        feed_dict={
            model.edges_labels: a,
            model.nodes_labels: x,
            model.node_features: f,
            model.training: False,
            model.variational: False,
        },
    )
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    y_true = e.flatten()
    y_pred = a.flatten()
    target_names = [
        str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()
    ]

    print("######## Classification Report ########\n")
    print(
        sk_classification_report(
            y_true,
            y_pred,
            labels=list(range(len(target_names))),
            target_names=target_names,
        )
    )

    print("######## Confusion Matrix ########\n")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    y_true = n.flatten()
    y_pred = x.flatten()
    target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

    print("######## Classification Report ########\n")
    print(
        sk_classification_report(
            y_true,
            y_pred,
            labels=list(range(len(target_names))),
            target_names=target_names,
        )
    )

    print("\n######## Confusion Matrix ########\n")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


def reconstructions(data, model, session, batch_dim=10, sample=False):
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    n, e = session.run(
        [model.nodes_gumbel_argmax, model.edges_gumbel_argmax]
        if sample
        else [model.nodes_argmax, model.edges_argmax],
        feed_dict={
            model.edges_labels: a,
            model.nodes_labels: x,
            model.node_features: f,
            model.training: False,
            model.variational: False,
        },
    )
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    m1 = np.array(
        [
            e if e is not None else Chem.RWMol()
            for e in [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
        ]
    )

    mols = np.vstack((m0, m1)).T.flatten()

    return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run(
        [model.nodes_gumbel_argmax, model.edges_gumbel_argmax]
        if sample
        else [model.nodes_argmax, model.edges_argmax],
        feed_dict={model.embeddings: embeddings, model.training: False},
    )
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols


def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {
        k: list(filter(lambda e: e is not None, v))
        for k, v in {
            "NP": MolecularMetrics.natural_product_scores(mols, norm=norm),
            "QED": MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
            "Solute": MolecularMetrics.water_octanol_partition_coefficient_scores(
                mols, norm=norm
            ),
            "SA": MolecularMetrics.synthetic_accessibility_score_scores(
                mols, norm=norm
            ),
            "diverse": MolecularMetrics.diversity_scores(mols, data),
            "drugcand": MolecularMetrics.drugcandidate_scores(mols, data),
        }.items()
    }

    m1 = {
        "valid": MolecularMetrics.valid_total_score(mols) * 100,
        "unique": MolecularMetrics.unique_total_score(mols) * 100,
        "novel": MolecularMetrics.novel_total_score(mols, data) * 100,
        "quality": MolecularMetrics.quality_scores(mols) * 100,
    }

    return m0, m1


def save_mol_img(mols, f_name="tmp.png", is_test=False):
    orig_f_name = f_name
    for a_mol in mols:
        try:
            if Chem.MolToSmiles(a_mol) is not None:
                print("Generating molecule")

                if is_test:
                    f_name = orig_f_name
                    f_split = f_name.split(".")
                    f_split[-1] = random_string() + "." + f_split[-1]
                    f_name = "".join(f_split)

                rdkit.Chem.Draw.MolToFile(a_mol, f_name)
                a_smi = Chem.MolToSmiles(a_mol)
                mol_graph = read_smiles(a_smi)

                break

                # if not is_test:
                #     break
        except Exception as e:
            print(e)
            continue
