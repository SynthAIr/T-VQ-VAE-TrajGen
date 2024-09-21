import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.decomposition import PCA

from timevqvae.evaluation import Metrics, MiniRocketTransform
from timevqvae.models import FidelityEnhancer
from timevqvae.trainers import Stage2
from timevqvae.utils import freeze, linear_warmup_cosine_annealingLR, log_image


class Stage3(L.LightningModule):
    def __init__(
        self,
        stage1_ckpt_fname,
        stage2_ckpt_fname,
        fcn_ckpt_fname,
        input_length: int,
        in_channels: int,
        n_classes: int,
        X_train: np.ndarray,
        X_test: np.ndarray,
        config: dict,
        device: torch.device,
        feature_extractor_type: str = "supervised_fcn",
    ):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.n_fft = config["VQ-VAE"]["n_fft"]
        self.tau_search_rng = config["fidelity_enhancer"]["tau_search_rng"]

        # FE
        self.fidelity_enhancer = FidelityEnhancer(
            input_length=input_length, in_channels=in_channels, config=config
        )

        # load the stage2 model
        stage2 = Stage2.load_from_checkpoint(
            stage2_ckpt_fname,
            stage1_ckpt_fname=stage1_ckpt_fname,
            fcn_ckpt_fname=fcn_ckpt_fname,
            input_length=input_length,
            in_channels=in_channels,
            n_classes=n_classes,
            X_train=X_train,
            X_test=X_test,
            config=config,
            device=device,
            feature_extractor_type=feature_extractor_type,
            map_location="cpu",
        )
        print("\nThe pretrained ExpStage2 is loaded.\n")
        freeze(stage2)
        stage2.eval()

        self.maskgit = stage2.maskgit
        self.encoder_l = self.maskgit.encoder_l
        self.decoder_l = self.maskgit.decoder_l
        self.vq_model_l = self.maskgit.vq_model_l
        self.encoder_h = self.maskgit.encoder_h
        self.decoder_h = self.maskgit.decoder_h
        self.vq_model_h = self.maskgit.vq_model_h

        self.minirocket = MiniRocketTransform(input_length)
        freeze(self.minirocket)
        self.percept_loss_weight = config["fidelity_enhancer"]["percept_loss_weight"]

        self.metrics = Metrics(
            fcn_ckpt_fname=fcn_ckpt_fname,
            input_length=input_length,
            in_channels=in_channels,
            n_classes=n_classes,
            batch_size=config["evaluation"]["batch_size"],
            X_train=X_train,
            X_test=X_test,
            device=device,
            feature_extractor_type=feature_extractor_type,
        )

    @torch.no_grad()
    def search_optimal_tau(
        self, X_train: np.ndarray, device, n_samples: int = 1024, batch_size: int = 32
    ) -> None:
        """
        must be run right after the instance is created.
        """
        maskgit = self.maskgit.to(device)

        n_iters = n_samples // batch_size + (0 if (n_samples % batch_size == 0) else 1)
        Xhat = torch.tensor([])
        for iter_idx in range(n_iters):
            s_l, s_h = maskgit.iterative_decoding(
                num=batch_size, device=device, class_index=None
            )
            x_new_l = maskgit.decode_token_ind_to_timeseries(s_l, "lf").cpu()
            x_new_h = maskgit.decode_token_ind_to_timeseries(s_h, "hf").cpu()
            x_new = x_new_l + x_new_h
            Xhat = torch.cat((Xhat, x_new))
        Xhat = Xhat.numpy().astype(float)
        Zhat = self.metrics.extract_feature_representations(Xhat)  # (b d)

        fids = []
        for i, tau in enumerate(self.tau_search_rng):
            print(
                f"searching optimal tau... ({round((i)/len(self.tau_search_rng) * 100, 1)}%)"
            )
            Xprime = []
            n_iters = X_train.shape[0] // batch_size + (
                0 if X_train.shape[0] % batch_size == 0 else 1
            )
            for i in range(n_iters):
                x = X_train[i * batch_size : (i + 1) * batch_size]
                x = torch.from_numpy(x).float().to(device)
                _, sprime_l = maskgit.encode_to_z_q(
                    x, self.encoder_l, self.vq_model_l, svq_temp=tau
                )  # (b n)
                _, sprime_h = maskgit.encode_to_z_q(
                    x, self.encoder_h, self.vq_model_h, svq_temp=tau
                )  # (b m)
                xprime_l = maskgit.decode_token_ind_to_timeseries(
                    sprime_l, "lf"
                )  # (b 1 l)
                xprime_h = maskgit.decode_token_ind_to_timeseries(
                    sprime_h, "hf"
                )  # (b 1 l)
                xprime = xprime_l + xprime_h  # (b c l)
                xprime = xprime.detach().cpu().numpy().astype(float)
                Xprime.append(xprime)
            Xprime = np.concatenate(Xprime)

            Z_prime = self.metrics.extract_feature_representations(Xprime)  # (b d)

            fid = self.metrics.fid_score(Zhat, Z_prime)
            fids.append(fid)
            print(f"tau:{tau} | fid:{round(fid,4)}")

            fig, axes = plt.subplots(3, 1, figsize=(4, 2 * 3))
            fig.suptitle(f"xhat vs x` (tau:{tau}, fid:{round(fid,4)})")
            axes[0].set_title("xhat")
            axes[0].plot(Xhat[:100, 0, :].T, color="C0", alpha=0.2)
            axes[1].set_title("x`")
            axes[1].plot(Xprime[:100, 0, :].T, color="C0", alpha=0.2)

            pca = PCA(n_components=2)
            Zhat_pca = pca.fit_transform(Zhat)
            Z_prime_pca = pca.transform(Z_prime)

            axes[2].scatter(Zhat_pca[:100, 0], Zhat_pca[:100, 1], alpha=0.2)
            axes[2].scatter(Z_prime_pca[:100, 0], Z_prime_pca[:100, 1], alpha=0.2)

            plt.tight_layout()
            fname = "Xhat_vs_Xprime.png"
            log_image(plt, fname)
            plt.close()
        print(
            "{tau:fid} :",
            {tau: round(float(fid), 4) for tau, fid in zip(self.tau_search_rng, fids)},
        )
        optimal_idx = np.argmin(fids)
        optimal_tau = self.tau_search_rng[optimal_idx]
        print("** optimal_tau **:", optimal_tau)
        self.fidelity_enhancer.tau = torch.tensor(optimal_tau).float()

    def _fidelity_enhancer_loss_fn(self, x, sprime_l, sprime_h):
        # s -> z -> x
        xprime_l = self.maskgit.decode_token_ind_to_timeseries(
            sprime_l, "lf"
        )  # (b 1 l)
        xprime_h = self.maskgit.decode_token_ind_to_timeseries(
            sprime_h, "hf"
        )  # (b 1 l)
        xprime = xprime_l + xprime_h  # (b c l)
        xprime = xprime.detach()

        xhat = self.fidelity_enhancer(xprime)
        recons_loss = F.l1_loss(xhat, x)

        fidelity_enhancer_loss = recons_loss
        return fidelity_enhancer_loss, (xprime, xhat)

    def _perceptual_loss_fn(self, x, xprime_R):
        if self.percept_loss_weight > 0:
            out = self.minirocket(torch.cat((xprime_R, x), dim=0))  # (2b d)
            out = rearrange(out, "(a b) d -> a b d", b=x.shape[0])  # (2 b d)
            percept_loss = self.percept_loss_weight * F.mse_loss(
                out[0, :, :], out[1, :, :]
            )
        else:
            percept_loss = 0.0
        return percept_loss

    def training_step(self, batch, batch_idx):
        self.eval()
        self.fidelity_enhancer.train()

        x, y = batch
        x = x.float()

        tau = self.fidelity_enhancer.tau.item()
        _, sprime_l = self.maskgit.encode_to_z_q(
            x, self.encoder_l, self.vq_model_l, svq_temp=tau
        )  # (b n)
        _, sprime_h = self.maskgit.encode_to_z_q(
            x, self.encoder_h, self.vq_model_h, svq_temp=tau
        )  # (b m)

        fidelity_enhancer_loss, (xprime, xprime_R) = self._fidelity_enhancer_loss_fn(
            x, sprime_l, sprime_h
        )
        percept_loss = self._perceptual_loss_fn(x, xprime_R)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss = fidelity_enhancer_loss + percept_loss
        loss_hist = {
            "loss": loss,
            "fidelity_enhancer_loss": fidelity_enhancer_loss,
            "percept_loss": percept_loss,
        }
        for k in loss_hist.keys():
            self.log(f"train/{k}", loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        x, y = batch
        x = x.float()

        tau = self.fidelity_enhancer.tau.item()
        _, sprime_l = self.maskgit.encode_to_z_q(
            x, self.encoder_l, self.vq_model_l, svq_temp=tau
        )  # (b n)
        _, sprime_h = self.maskgit.encode_to_z_q(
            x, self.encoder_h, self.vq_model_h, svq_temp=tau
        )  # (b m)

        fidelity_enhancer_loss, (xprime, xprime_R) = self._fidelity_enhancer_loss_fn(
            x, sprime_l, sprime_h
        )
        percept_loss = self._perceptual_loss_fn(x, xprime_R)

        # log
        loss = fidelity_enhancer_loss + percept_loss
        loss_hist = {
            "loss": loss,
            "fidelity_enhancer_loss": fidelity_enhancer_loss,
            "percept_loss": percept_loss,
        }
        for k in loss_hist.keys():
            self.log(f"val/{k}", loss_hist[k])

        # maskgit sampling
        if batch_idx == 0:
            class_index = None

            # unconditional sampling
            num = 1024
            xhat_l, xhat_h, xhat = self.metrics.sample(
                self.maskgit, x.device, num, "unconditional", class_index=class_index
            )
            xhat_R = self.fidelity_enhancer(xhat.to(x.device)).detach().cpu().numpy()

            b = 0
            n_figs = 9
            fig, axes = plt.subplots(n_figs, 1, figsize=(4, 2 * n_figs))
            fig.suptitle(f"step-{self.global_step}; class idx: {class_index}")

            axes[0].set_title(r"$\hat{x}_l$")
            axes[0].plot(xhat_l[b, 0, :])
            axes[1].set_title(r"$\hat{x}_h$")
            axes[1].plot(xhat_h[b, 0, :])
            axes[2].set_title(r"$\hat{x}$")
            axes[2].plot(xhat[b, 0, :])
            axes[3].set_title(r"FE($\hat{x}$) = $\hat{x}_R$")
            axes[3].plot(xhat_R[b, 0, :])

            x = x.cpu().numpy()
            xprime = xprime.cpu().numpy()
            xprime_R = xprime_R.cpu().numpy()
            xhat = xhat.cpu().numpy()
            b_ = np.random.randint(0, x.shape[0])
            axes[4].set_title(r"$x$ vs FE($x^\prime$)")
            axes[4].plot(x[b_, 0, :], alpha=0.7)
            axes[4].plot(xprime_R[b_, 0, :], alpha=0.7)

            axes[5].set_title(r"$x^\prime$ vs FE($x^\prime$)")
            axes[5].plot(xprime[b_, 0, :], alpha=0.7)
            axes[5].plot(xprime_R[b_, 0, :], alpha=0.7)

            axes[6].set_title(r"$x$")
            axes[6].plot(x[b_, 0, :])

            axes[7].set_title(rf"$x^\prime$ ($\tau$={round(tau, 5)})")
            axes[7].plot(xprime[b_, 0, :])

            axes[8].set_title(r"FE($x^\prime$)")
            axes[8].plot(xprime_R[b_, 0, :])

            for ax in axes:
                ax.set_ylim(-4, 4)
            plt.tight_layout()
            # self.logger.log_image(key='unconditionally generated sample', images=[wandb.Image(plt),])
            fname = "unconditionally_generated_sample.png"
            log_image(plt, fname)
            plt.close()

            # log the evaluation metrics

            zhat = self.metrics.z_gen_fn(xhat)
            fid_test_gen = self.metrics.fid_score(self.metrics.z_test, zhat)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat)
            self.log("running_metrics/FID", fid_test_gen)
            self.log("running_metrics/MDD", mdd)
            self.log("running_metrics/ACD", acd)
            self.log("running_metrics/SD", sd)
            self.log("running_metrics/KD", kd)

            zhat_R = self.metrics.z_gen_fn(xhat_R)
            fid_test_gen_fe = self.metrics.fid_score(self.metrics.z_test, zhat_R)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat_R)
            self.log("running_metrics/FID with FE", fid_test_gen_fe)
            self.log("running_metrics/MDD with FE", mdd)
            self.log("running_metrics/ACD with FE", acd)
            self.log("running_metrics/SD with FE", sd)
            self.log("running_metrics/KD with FE", kd)

            z_prime = self.metrics.z_gen_fn(xprime)
            fid_x_test_x_prime = self.metrics.fid_score(self.metrics.z_test, z_prime)
            # fid_train_x_prime, fid_test_x_prime = self.metrics.fid_score(xprime)
            mdd, acd, sd, kd = self.metrics.stat_metrics(x, xprime)
            self.log("running_metrics/FID_x_xhat", fid_x_test_x_prime)
            self.log("running_metrics/MDD_x_xhat", mdd)
            self.log("running_metrics/ACD_x_xhat", acd)
            self.log("running_metrics/SD_x_xhat", sd)
            self.log("running_metrics/KD_x_xhat", kd)

            plt.figure(figsize=(4, 4))
            plt.title(f"step-{self.global_step}")
            labels = ["Z_test", "Zhat_R"]
            pca = PCA(n_components=2, random_state=0)
            for i, (Z, label) in enumerate(zip([self.metrics.z_test, zhat_R], labels)):
                ind = np.random.choice(range(Z.shape[0]), size=num, replace=True)
                Z_embed = pca.fit_transform(Z[ind]) if i == 0 else pca.transform(Z[ind])
                plt.scatter(Z_embed[:, 0], Z_embed[:, 1], alpha=0.1, label=label)
            plt.legend(loc="upper right")
            plt.tight_layout()
            fname = f"PCA_on_Z_{labels}.png"
            log_image(plt, fname)
            plt.close()

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config["exp_params"]["lr"])
        scheduler = linear_warmup_cosine_annealingLR(
            opt,
            self.config["trainer_params"]["max_steps"]["stage3"],
            self.config["exp_params"]["linear_warmup_rate"],
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}
