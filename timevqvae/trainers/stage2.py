import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn

from timevqvae.evaluation import Metrics
from timevqvae.models import MaskGIT
from timevqvae.utils import linear_warmup_cosine_annealingLR, log_image


class Stage2(L.LightningModule):
    def __init__(
        self,
        stage1_ckpt_fname: str,
        fcn_ckpt_fname: str,
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
        self.maskgit = MaskGIT(
            stage1_ckpt_fname=stage1_ckpt_fname,
            input_length=input_length,
            in_channels=in_channels,
            config=config,
            n_classes=n_classes,
            **config["MaskGIT"],
        )

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

    def training_step(self, batch, batch_idx):
        x, y = batch

        mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self.maskgit(x, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": mask_pred_loss,
            "mask_pred_loss": mask_pred_loss,
            "mask_pred_loss_l": mask_pred_loss_l,
            "mask_pred_loss_h": mask_pred_loss_h,
        }
        for k in loss_hist.keys():
            self.log(f"train/{k}", loss_hist[k])

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        x, y = batch

        mask_pred_loss, (mask_pred_loss_l, mask_pred_loss_h) = self.maskgit(x, y)

        # log
        loss_hist = {
            "loss": mask_pred_loss,
            "mask_pred_loss": mask_pred_loss,
            "mask_pred_loss_l": mask_pred_loss_l,
            "mask_pred_loss_h": mask_pred_loss_h,
        }
        for k in loss_hist.keys():
            self.log(f"val/{k}", loss_hist[k])

        # maskgit sampling & evaluation
        if batch_idx == 0 and (self.training == False):
            print("computing evaluation metrices...")
            self.maskgit.eval()

            n_samples = 1024
            xhat_l, xhat_h, xhat = self.metrics.sample(
                self.maskgit, x.device, n_samples, "unconditional", class_index=None
            )

            self._visualize_generated_timeseries(xhat_l, xhat_h, xhat)

            # compute metrics
            xhat = xhat.numpy()
            zhat = self.metrics.z_gen_fn(xhat)
            fid_test_gen = self.metrics.fid_score(self.metrics.z_test, zhat)
            mdd, acd, sd, kd = self.metrics.stat_metrics(self.metrics.X_test, xhat)
            self.log("running_metrics/FID", fid_test_gen)
            self.log("running_metrics/MDD", mdd)
            self.log("running_metrics/ACD", acd)
            self.log("running_metrics/SD", sd)
            self.log("running_metrics/KD", kd)

        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.config["exp_params"]["lr"])
        scheduler = linear_warmup_cosine_annealingLR(
            opt,
            self.config["trainer_params"]["max_steps"]["stage2"],
            self.config["exp_params"]["linear_warmup_rate"],
        )
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def _visualize_generated_timeseries(self, xhat_l, xhat_h, xhat):
        b = 0
        c = np.random.randint(0, xhat.shape[1])
        n_rows = 3
        fig, axes = plt.subplots(n_rows, 1, figsize=(4, 2 * n_rows))
        fig.suptitle(
            f"step-{self.global_step} | channel idx:{c} \n unconditional sampling"
        )
        axes = axes.flatten()
        axes[0].set_title(r"$\hat{x}_l$ (LF)")
        axes[0].plot(xhat_l[b, c, :])
        axes[1].set_title(r"$\hat{x}_h$ (HF)")
        axes[1].plot(xhat_h[b, c, :])
        axes[2].set_title(r"$\hat{x}$ (LF+HF)")
        axes[2].plot(xhat[b, c, :])
        for ax in axes:
            ax.set_ylim(-4, 4)
        plt.tight_layout()
        log_image(plt, "generated_sample.png")
        plt.close()
