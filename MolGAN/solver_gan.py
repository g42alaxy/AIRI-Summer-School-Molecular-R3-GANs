import datetime
import os
import time
from collections import defaultdict
import logging

import numpy as np
import torch
import torch.nn.functional as F
from data.sparse_molecular_dataset import SparseMolecularDataset
from models_gan import Discriminator, Generator
from utils import MolecularMetrics, all_scores, save_mol_img


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, config, log=None):
        """Initialize configurations."""

        # Log
        self.log = True

        print("config", config)

        # Data loader.
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Model configurations.
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.la = config.lambda_wgan
        self.la_gp = config.lambda_gp
        self.post_method = config.post_method
        self.eval_freq = config.eval_freq
        self.n_molecules_validation = config.n_molecules_validation
        self.no_rl_epochs = config.no_rl_epochs
        # self.metric = "validity,qed"
        self.metric = config.metric
        # Batch-level discriminator flag
        self.batch_discriminator = getattr(config, "batch_discriminator", False)

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = len(self.data) // self.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        if self.la > 0:
            self.n_critic = config.n_critic
        else:
            self.n_critic = 1
        self.resume_epoch = config.resume_epoch

        # Training or testing.
        self.mode = config.mode

        # Miscellaneous.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        # Directories.
        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path

        # Step size.
        self.model_save_step = config.model_save_step

        # Build the model.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(
            self.g_conv_dim,
            self.z_dim,
            self.data.vertexes,
            self.data.bond_num_types,
            self.data.atom_num_types,
            self.dropout,
        )
        self.D = Discriminator(
            self.d_conv_dim,
            self.m_dim,
            self.b_dim - 1,
            dropout_rate=self.dropout,
            batch_discriminator=self.batch_discriminator,
        )
        self.V = Discriminator(
            self.d_conv_dim,
            self.m_dim,
            self.b_dim - 1,
            dropout_rate=self.dropout,
            batch_discriminator=self.batch_discriminator,
        )

        # ------------------------------------------------------------------
        # TF implementation used Adam on all nets – mirror that here.
        # ------------------------------------------------------------------
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=self.g_lr, betas=(0.0, 0.9)
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=self.d_lr, betas=(0.0, 0.9)
        )
        self.v_optimizer = torch.optim.Adam(
            self.V.parameters(), lr=self.g_lr, betas=(0.0, 0.9)
        )
        self.print_network(self.G, "G", self.log)
        self.print_network(self.D, "D", self.log)
        self.print_network(self.V, "V", self.log)

        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            logging.info(model)
            logging.info(name)
            logging.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print(f"Loading the trained models from step {resume_iters}...")
        G_path = os.path.join(self.model_dir_path, f"{resume_iters}-G.ckpt")
        D_path = os.path.join(self.model_dir_path, f"{resume_iters}-D.ckpt")
        V_path = os.path.join(self.model_dir_path, f"{resume_iters}-V.ckpt")
        self.G.load_state_dict(
            torch.load(G_path, map_location=lambda storage, loc: storage)
        )
        self.D.load_state_dict(
            torch.load(D_path, map_location=lambda storage, loc: storage)
        )
        self.V.load_state_dict(
            torch.load(V_path, map_location=lambda storage, loc: storage)
        )

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones_like(y)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.norm(dydx, dim=1)
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        return F.one_hot(labels, num_classes=dim).float().to(self.device)

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.z_dim, device=self.device)

    @staticmethod
    def postprocess(inputs, method, temperature=1.0):
        def listify(x):
            return x if isinstance(x, (list, tuple)) else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == "soft_gumbel":
            softmax = [
                F.gumbel_softmax(
                    e_logits.contiguous().reshape(-1, e_logits.size(-1)) / temperature,
                    hard=False,
                ).reshape(e_logits.size())
                for e_logits in listify(inputs)
            ]
        elif method == "hard_gumbel":
            softmax = [
                F.gumbel_softmax(
                    e_logits.contiguous().reshape(-1, e_logits.size(-1)) / temperature,
                    hard=True,
                ).reshape(e_logits.size())
                for e_logits in listify(inputs)
            ]
        else:
            softmax = [
                F.softmax(e_logits / temperature, -1) for e_logits in listify(inputs)
            ]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols):
        """Calculate reward scores for molecules based on configured metrics.

        Args:
            mols: List of molecule objects

        Returns:
            numpy.ndarray: Reward scores reshaped to (-1, 1)
        """
        # Define metric calculation mapping
        metric_calculators = {
            "np": lambda m: MolecularMetrics.natural_product_scores(m, norm=True),
            "logp": lambda m: MolecularMetrics.water_octanol_partition_coefficient_scores(
                m, norm=True
            ),
            "sas": lambda m: MolecularMetrics.synthetic_accessibility_score_scores(
                m, norm=True
            ),
            "qed": lambda m: MolecularMetrics.quantitative_estimation_druglikeness_scores(
                m, norm=True
            ),
            "novelty": lambda m: MolecularMetrics.novel_scores(m, self.data),
            "dc": lambda m: MolecularMetrics.drugcandidate_scores(m, self.data),
            "unique": lambda m: MolecularMetrics.unique_scores(m),
            "diversity": lambda m: MolecularMetrics.diversity_scores(m, self.data),
            "validity": lambda m: MolecularMetrics.valid_scores(m),
        }

        # Parse metrics to evaluate
        metrics_to_use = self._get_metrics_list()

        # Calculate reward as product of all metric scores
        reward_score = 1.0
        for metric in metrics_to_use:
            if metric not in metric_calculators:
                raise RuntimeError(f"{metric} is not defined as a metric")
            reward_score = reward_score * metric_calculators[metric](mols)

        return reward_score.reshape(-1, 1)

    def _get_metrics_list(self):
        """Parse the metrics string into a list of individual metrics."""
        if self.metric == "all":
            return ["logp", "sas", "qed", "unique"]
        return [m.strip() for m in self.metric.split(",")]

    def train_and_validate(self):
        self.start_time = time.time()

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.resume_epoch is not None:
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)

        # Start training.
        if self.mode == "train":
            self.validate_epoch(epoch=start_epoch, val_n=self.n_molecules_validation)
            print("Start training...")
            for i in range(start_epoch, self.num_epochs):
                self.train_epoch(epoch=i)
                if (i + 1) % self.eval_freq == 0:
                    self.validate_epoch(epoch=i, val_n=self.n_molecules_validation)
        elif self.mode == "test":
            assert self.resume_epoch is not None
            self.validate_epoch(epoch=start_epoch)
        else:
            raise NotImplementedError

    def _process_outputs(self, n_hat, e_hat, method):
        edges_hard, nodes_hard = self.postprocess((e_hat, n_hat), method)
        edges_hard = torch.max(edges_hard, -1)[1]
        nodes_hard = torch.max(nodes_hard, -1)[1]

        mols = [
            self.data.matrices2mol(
                n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True
            )
            for e_, n_ in zip(edges_hard, nodes_hard)
        ]
        return mols

    def get_gen_mols(self, n_hat, e_hat, method):
        return self._process_outputs(n_hat, e_hat, method)

    def get_reward(self, n_hat, e_hat, method):
        mols = self._process_outputs(n_hat, e_hat, method)
        return torch.from_numpy(self.reward(mols)).to(self.device)

    def save_checkpoints(self, epoch_i):
        G_path = os.path.join(self.model_dir_path, "{}-G.ckpt".format(epoch_i + 1))
        D_path = os.path.join(self.model_dir_path, "{}-D.ckpt".format(epoch_i + 1))
        V_path = os.path.join(self.model_dir_path, "{}-V.ckpt".format(epoch_i + 1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.V.state_dict(), V_path)
        print("Saved model checkpoints into {}...".format(self.model_dir_path))
        if self.log is not None:
            logging.info(
                "Saved model checkpoints into {}...".format(self.model_dir_path)
            )

    def train_epoch(self, epoch):
        # Recordings
        losses = defaultdict(list)

        for a_step in range(self.num_steps):
            mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
            z = self.sample_z(self.batch_size)

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            a = torch.from_numpy(a).to(self.device).long()  # Adjacency.
            x = torch.from_numpy(x).to(self.device).long()  # Nodes.
            a_tensor = self.label2onehot(a, self.b_dim)
            x_tensor = self.label2onehot(x, self.m_dim)

            # Current steps
            cur_step = self.num_steps * epoch + a_step
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute losses with real inputs.
            logits_real, features_real = self.D(a_tensor, None, x_tensor)

            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess(
                (edges_logits, nodes_logits), self.post_method
            )
            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)

            # Compute losses for gradient penalty (WGAN-GP). Use *independent* eps masks
            # for adjacency and node tensors to reduce correlation and improve stability.
            eps_e = torch.rand(a_tensor.size(0), 1, 1, 1, device=self.device)
            eps_n = torch.rand(x_tensor.size(0), 1, 1, device=self.device)

            x_int0 = (
                (eps_e * a_tensor + (1 - eps_e) * edges_hat)
                .detach()
                .requires_grad_(True)
            )
            x_int1 = (
                (eps_n * x_tensor + (1 - eps_n) * nodes_hat)
                .detach()
                .requires_grad_(True)
            )
            grad0, grad1 = self.D(x_int0, None, x_int1)
            grad_penalty = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(
                grad1, x_int1
            )

            d_loss_real = torch.mean(logits_real)
            d_loss_fake = torch.mean(logits_fake)
            loss_D = -d_loss_real + d_loss_fake + self.la_gp * grad_penalty

            losses["l_D/R"].append(d_loss_real.item())
            losses["l_D/F"].append(d_loss_fake.item())
            losses["l_D"].append(loss_D.item())

            self.reset_grad()
            loss_D.backward()
            self.d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator & value net                   #
            # =================================================================================== #
            
            # (re-sample latent for generator step – keeps it consistent with TF code)
            z = self.sample_z(self.batch_size)
            edges_logits, nodes_logits = self.G(z)
            (edges_hat, nodes_hat) = self.postprocess(
                (edges_logits, nodes_logits), self.post_method
            )
            
            # Detach these tensors to prevent them from being part of the generator's graph
            edges_hat_detached = edges_hat.detach()
            nodes_hat_detached = nodes_hat.detach()
            
            logits_fake, _ = self.D(edges_hat, None, nodes_hat)
            logits_real, _ = self.D(a_tensor.detach(), None, x_tensor.detach())
            
            # -------------------- Generator adversarial / feature matching loss -----------------
            loss_G_adv = -torch.mean(logits_fake)
            
            # -------------------- RL & Value losses ---------------------------------------------
            la_curr = 1.0 if epoch < self.no_rl_epochs else self.la
            if la_curr < 1.0:
                value_logit_real, _ = self.V(
                    a_tensor, None, x_tensor, torch.sigmoid
                )
                # Use the detached versions for value network computation
                value_logit_fake, _ = self.V(edges_hat.detach(), None, nodes_hat.detach(), torch.sigmoid)
            
                reward_r = torch.from_numpy(self.reward(mols)).to(self.device)
                reward_f = self.get_reward(nodes_hat.detach(), edges_hat.detach(), self.post_method)
            
                loss_V = torch.mean(
                    (value_logit_real - reward_r) ** 2
                    + (value_logit_fake - reward_f) ** 2
                )
                loss_RL = -torch.mean(value_logit_fake)
            
                alpha = torch.abs(loss_G_adv.detach() / loss_RL.detach())
                gen_total_loss = la_curr * loss_G_adv + (1 - la_curr) * alpha * loss_RL
            else:
                value_logit_real, _ = self.V(
                    a_tensor, None, x_tensor, torch.sigmoid
                )
                loss_V = torch.zeros(1, device=self.device).float()
                loss_RL = torch.zeros(1, device=self.device).float()
                gen_total_loss = loss_G_adv
            
            if cur_step % self.n_critic == 0:
                # -------------------- Generator update --------------------------------------
                self.g_optimizer.zero_grad()
                gen_total_loss.backward(retain_graph=True)
                self.g_optimizer.step()
            
                # -------------------- Value network update (only if lambda_wgan < 1) --------
                if self.la < 1.0:
                    self.v_optimizer.zero_grad()
                    # Don't need to retain graph anymore since we're using detached tensors
                    loss_V.backward()
                    self.v_optimizer.step()

            # Record.
            losses["l_G"].append(loss_G_adv.item())
            losses["l_RL"].append(loss_RL.item())
            losses["l_V"].append(loss_V.item())

        # Save checkpoints.
        if self.mode == "train" and (epoch + 1) % self.model_save_step == 0:
            self.save_checkpoints(epoch_i=epoch)

        # Print out training information.
        elapsed = time.time() - self.start_time
        et_str = str(datetime.timedelta(seconds=elapsed))[:-7]
        header = f"Elapsed [{et_str}], Iteration [{epoch + 1}/{self.num_epochs}]:"

        loss_str = ", ".join(
            f"{tag}: {np.mean(vals):.2f}" for tag, vals in losses.items()
        )
        print(f"{header}\n{loss_str}")
        if self.log:
            logging.info(f"{header}\n{loss_str}")

        return losses

    @torch.no_grad()
    def validate_epoch(self, epoch, val_n=None):
        molecules = []
        self.G.eval()

        if val_n is None:
            val_n = self.data.validation_count
        val_steps = val_n // self.batch_size

        metrics_acc = defaultdict(list)

        for _ in range(val_steps + 1):
            # advance the validation counter (we ignore the outputs)
            z = self.sample_z(self.batch_size)
            edges_logits, nodes_logits = self.G(z)
            molecules.extend(
                self.get_gen_mols(nodes_logits, edges_logits, self.post_method)
            )

        m0, m1 = all_scores(molecules, self.data, norm=True)
        for k, v in m1.items():
            metrics_acc[k].append(v)
        for k, v in m0.items():
            metrics_acc[k].append(np.array(v)[np.nonzero(v)].mean())

        # average & report
        # report = ", ".join(
        #     f"{tag}: {sum(vals) / len(vals):.4f}" for tag, vals in metrics_acc.items()
        # )
        score_str = ", ".join(
            f"{tag}: {np.mean(vals):.2f}" for tag, vals in metrics_acc.items()
        )
        msg = f"[Val][Epoch {epoch + 1}/{self.num_epochs}] {score_str}"
        print(msg)
        if self.log:
            logging.info(msg)

            # Saving molecule images.
            # mol_f_name = os.path.join(self.img_dir_path, "mol-{}.png".format(epoch_i))
            # save_mol_img(mols, mol_f_name, is_test=self.mode == "test")
