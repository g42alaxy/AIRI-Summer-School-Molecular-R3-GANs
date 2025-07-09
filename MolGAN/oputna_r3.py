import logging
import os
import optuna
import wandb

import numpy as np

from args import get_GAN_config
from rdkit import RDLogger
from solver_r3gan import Solver
from torch.backends import cudnn
from util_dir.utils_io import get_date_postfix
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Remove flooding logs.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def collect_solver(config, wandb_=False):
    # For fast training.
    cudnn.benchmark = True

    # Timestamp
    if config.mode == "train":
        config.saving_dir = os.path.join(config.saving_dir, get_date_postfix())
        config.log_dir_path = os.path.join(config.saving_dir, "log_dir")
        config.model_dir_path = os.path.join(config.saving_dir, "model_dir")
        config.img_dir_path = os.path.join(config.saving_dir, "img_dir")
    else:
        a_test_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir)
        config.log_dir_path = os.path.join(
            config.saving_dir, "post_test", a_test_time, "log_dir"
        )
        config.model_dir_path = os.path.join(config.saving_dir, "model_dir")
        config.img_dir_path = os.path.join(
            config.saving_dir, "post_test", a_test_time, "img_dir"
        )

    # Create directories if not exist.
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.model_dir_path):
        os.makedirs(config.model_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    # Logger
    if config.mode == "train":
        log_p_name = os.path.join(
            config.log_dir_path, get_date_postfix() + "_logger.log"
        )
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)

    # Solver for training and testing StarGAN.
    if config.mode == "train":
        solver = Solver(config, logging, wandb_=wandb_)
    elif config.mode == "test":
        solver = Solver(config, wandb_=wandb_)
    else:
        raise NotImplementedError

    return solver


def objective(trial):    
    config = get_GAN_config()
    config.n_molecules_validation = 1000
    config.num_epochs  = 50
    config.post_method = "soft_gumbel"
    config.eval_freq   = 5

    params = {
        'g_lr': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        'd_lr': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        'lambda_gp': trial.suggest_float('lambda_gp', 1e-2, 2e1, log=True),
    }

    #real config for model assemble 
    config.lambda_gp =  params['lambda_gp']
    config.g_lr = params['g_lr']
    config.d_lr = params['d_lr']

    #wandb config init which includes only searchable parameters (as it's just easier to track)
    with wandb.init(
        project="R3-MolGAN-optuna-new",
        config=params,
        reinit=True,
        tags=["optuna_trial"],
        group="hyperparameter_search",
        name=f"trial-{trial.number}"
    ):

        solver = collect_solver(config, wandb_=True)
        quality_scores = solver.train_and_validate()

        wandb.log({"best_quality": np.max(quality_scores)})
        #wandb_run.finish()

    return np.max(quality_scores)
    #print(config)

#objective()    

class ResultLogger:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write("trial_number,value,params,datetime\n")
    
    def __call__(self, study, trial):
        with open(self.filename, 'a') as f:
            f.write(f"{trial.number},{trial.value},{trial.params},{datetime.now()}\n")

if __name__ == "__main__":
    logger = ResultLogger('optuna__r3_logs.log')
    
    study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
    )
        
    study.optimize(objective, n_trials=200, timeout=5*3600, callbacks=[logger])