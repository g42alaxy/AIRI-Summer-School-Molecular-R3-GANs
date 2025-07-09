<h1 align="center">
    <img width="300" height="auto" src="https://github.com/user-attachments/assets/9d294e49-cd54-4dfd-88c4-437740cb4d4d" />
    <br>
    AIRI Summer School: R3 GANs in Molecular domain
    <br>
</h1>

<div align="center">
  
[![Presentation](https://img.shields.io/badge/🖼️_Presentation-Google_Slides-orange)](https://docs.google.com/presentation/d/1d8HeLQwCl_oa7WI8OVOoWR22C9a3WJ8diN4sRolWQyE/edit?usp=sharing)
[![Report](https://img.shields.io/badge/📄_Report-Overleaf-green)]([https://www.overleaf.com/read/fvbnfvhqvfxb#345093](https://www.overleaf.com/read/ytrgfcyfbdxx#16d5ff))

</div>

### 1. R3-MolGAN

##### 1.0 Environment setup:

##### 1.1 Dateset preparation:

By default MolGAN is trained either on entire QM9 dataset, which is accessible via `download_dataset.sh` in `MolGAN/data/`, or on it's 5k-subset which is already prepared as `MolGAN/data/qm9_5k.smi`. Any `*.sdf` or `*.smi` dataset should be converted to MolGAN's graph representations with `sparse_molecular_dataset.py` 

### 2. R3-LatentGAN, trained on [MOSES](https://github.com/molecularsets/moses)

##### 2.0 Environment setup:

LatentGAN requires separate environment, which can be installed as [MOSES](https://github.com/molecularsets/moses) benchmark installation instruction suggests. To setup the R3-LatentGAN, after [MOSES](https://github.com/molecularsets/moses) is installed, simply put everything from `latentgan/` into `moses/` folder. 

##### 2.1 Training of R3-LatentGAN:
```shell
python3 scripts/train.py r3latentgan --model_save=checkpoints_r3latentgan  --config_save=config_r3latentgan  --vocab_save=vocab_r3latentgan --log_file=log_r3latentgan --device=cuda:0
```

##### 2.2 Sampling with R3-LatentGAN:
Following examples samples with checkpoint after `300` epochs of training
```shell
python3 scripts/sample.py r3latentgan --model_load=checkpoints_r3latent_300.pt --config=config_r3latentgan --n_samples=15000 --gen_save=generated_molecules.smi --device=cuda:0 --vocab_load=vocab_r3latentgan
```

##### 2.3 Samples evaluation R3-LatentGAN:
- Evaluation with  [MOSES](https://github.com/molecularsets/moses) metrics
  
<table>
  <tr>
    <td width="100%" style="background-color: #FFEBEE; border-left: 4px solid #F44336; padding: 12px; color: #C62828;">
      <strong>Note:</strong> In MOSES, QED and other chemical metrics correspond to their W1-Wasserstein distance from training ones.
    </td>
  </tr>
</table>


```shell
python3 scripts/sample.py r3latentgan --model_load=checkpoints_r3latent_300.pt --config=config_r3latentgan --n_samples=15000 --gen_save=generated_molecules.smi --device=cuda:0 --vocab_load=vocab_r3latentgan
```

- Evaluation with self-implemented MolGAN-like metrics (including new quality metrics)
```shell
python3 assess_chem.py generated_molecules.smi
```

### 3. Baseline model: [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules), trained on [nablaDFT](https://github.com/AIRI-Institute/nablaDFT/).
To reproduce our results:

```shell
conda create -n baseline_env --file baseline_environment.yml
conda activate baseline_env
git clone https://github.com/ehoogeboom/e3_diffusion_for_molecules.git
wget 'https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/energy_databases/train_100k_v2_formation_energy_w_forces.db'
wget 'https://a002dlils-kadurin-nabladft.obs.ru-moscow-1.hc.sbercloud.ru/data/nablaDFTv2/summary.csv.gz'
cp scripts/diff_nabla.py e3_diffusion_for_molecules/
cd e3_diffusion_for_molecules
pip install .
python diff_nabla.py
```
