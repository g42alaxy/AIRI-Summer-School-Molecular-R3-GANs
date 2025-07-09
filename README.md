<h1 align="center">
    <img width="300" height="auto" src="https://github.com/user-attachments/assets/9d294e49-cd54-4dfd-88c4-437740cb4d4d" />
    <br>
    AIRI Summer School: R3 GANs in Molecular domain
    <br>
</h1>

Baseline model: [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules), trained on [nablaDFT](https://github.com/AIRI-Institute/nablaDFT/).
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
