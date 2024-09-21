"""
FID, IS, JS divergence.
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from timevqvae.evaluation import (apply_kernels, auto_correlation_difference,
                                  calculate_fid, calculate_inception_score,
                                  generate_kernels, kurtosis_difference,
                                  load_pretrained_FCN,
                                  marginal_distribution_difference,
                                  skewness_difference)
from timevqvae.models import FidelityEnhancer
from timevqvae.trainers import Stage2
from timevqvae.utils import (conditional_sample, log_image, remove_outliers,
                             unconditional_sample)


class TrainedModelSampler(nn.Module):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """

    def __init__(
        self,
        stage1_ckpt_fname,
        stage2_ckpt_fname,
        stage3_ckpt_fname,
        fcn_ckpt_fname,
        input_length: int,
        in_channels: int,
        n_classes: int,
        batch_size: int,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        device: torch.device,
        config: dict,
        use_fidelity_enhancer: bool = True,
        feature_extractor_type: str = "supervised_fcn",
        rocket_num_kernels: int = 1000,
        do_evaluate: bool = True,
    ):
        super().__init__()

        self.device = device
        self.config = config
        self.X_train, self.Y_train, self.X_test, self.Y_test = (
            X_train,
            Y_train,
            X_test,
            Y_test,
        )
        self.batch_size = batch_size

        self.feature_extractor_type = feature_extractor_type
        assert feature_extractor_type in [
            "supervised_fcn",
            "rocket",
        ], "unavailable feature extractor type."

        # load the stage2 model
        self.stage2 = Stage2.load_from_checkpoint(
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
        self.stage2.eval()
        self.maskgit = self.stage2.maskgit
        self.stage1 = self.stage2.maskgit.stage1

        # load the fidelity enhancer
        if use_fidelity_enhancer:
            self.fidelity_enhancer = FidelityEnhancer(
                input_length=input_length, in_channels=in_channels, config=config
            )

            ckpt = torch.load(stage3_ckpt_fname, map_location="cpu")
            fidelity_enhancer_state_dict = {
                k.replace("fidelity_enhancer.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("fidelity_enhancer.")
            }
            self.fidelity_enhancer.load_state_dict(fidelity_enhancer_state_dict)
        else:
            self.fidelity_enhancer = nn.Identity()

        if do_evaluate:
            self.fcn = load_pretrained_FCN(
                ckpt_fname=fcn_ckpt_fname, in_channels=in_channels, n_classes=n_classes
            ).to(self.device)
            self.fcn.eval()
            if feature_extractor_type == "rocket":
                self.rocket_kernels = generate_kernels(
                    input_length, num_kernels=rocket_num_kernels
                )

            self.ts_len = self.X_train.shape[-1]  # time series length
            self.n_classes = len(np.unique(self.Y_train))

            # fit PCA on a training set
            self.pca = PCA(n_components=2, random_state=0)
            self.z_train = self.compute_z("train")
            self.z_test = self.compute_z("test")

            z_test = remove_outliers(
                self.z_test
            )  # only used to fit pca because `def fid_score` already contains `remove_outliers`
            z_transform_pca = self.pca.fit_transform(z_test)

            self.xmin_pca, self.xmax_pca = np.min(z_transform_pca[:, 0]), np.max(
                z_transform_pca[:, 0]
            )
            self.ymin_pca, self.ymax_pca = np.min(z_transform_pca[:, 1]), np.max(
                z_transform_pca[:, 1]
            )

    @torch.no_grad()
    def sample(self, n_samples: int, kind: str, class_index: Union[int, None] = None):
        assert kind in ["unconditional", "conditional"]

        # sampling
        if kind == "unconditional":
            x_new_l, x_new_h, x_new = unconditional_sample(
                self.maskgit, n_samples, self.device, batch_size=self.batch_size
            )  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == "conditional":
            x_new_l, x_new_h, x_new = conditional_sample(
                self.maskgit, n_samples, self.device, class_index, self.batch_size
            )  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError

        # FE
        num_batches = x_new.shape[0] // self.batch_size + (
            1 if x_new.shape[0] % self.batch_size != 0 else 0
        )
        X_new_R = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            mini_batch = x_new[start_idx:end_idx]
            x_new_R = self.fidelity_enhancer(mini_batch.to(self.device)).cpu()
            X_new_R.append(x_new_R)
        X_new_R = torch.cat(X_new_R)

        return (x_new_l, x_new_h, x_new), X_new_R

    def _extract_feature_representations(self, x: np.ndarray):
        """
        x: (b 1 l)
        """
        if self.feature_extractor_type == "supervised_fcn":
            z = (
                self.fcn(
                    torch.from_numpy(x).float().to(self.device),
                    return_feature_vector=True,
                )
                .cpu()
                .detach()
                .numpy()
            )  # (b d)
        elif self.feature_extractor_type == "rocket":
            x = x[:, 0, :]  # (b l)
            x = x.astype(np.float64)  # Convert to float64 for rocket
            z = apply_kernels(x, self.rocket_kernels)
            z = F.normalize(torch.from_numpy(z), p=2, dim=1).numpy()
        else:
            raise ValueError
        return z

    def compute_z_rec(self, kind: str):
        """
        compute representations of X_rec
        """
        assert kind in ["train", "test"]
        if kind == "train":
            X = self.X_train  # (b 1 l)
        elif kind == "test":
            X = self.X_test  # (b 1 l)
        else:
            raise ValueError

        n_samples = X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test`
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x = X[s]  # (b 1 l)
            x = torch.from_numpy(x).float().to(self.device)
            x_rec = (
                self.stage1.forward(batch=(x, None), batch_idx=-1, return_x_rec=True)
                .cpu()
                .detach()
                .numpy()
                .astype(float)
            )  # (b 1 l)
            z_t = self._extract_feature_representations(x_rec)
            zs.append(z_t)
        zs = np.concatenate(zs, axis=0)
        return zs

    @torch.no_grad()
    def compute_z_svq(self, kind: str):
        """
        compute representations of X', a stochastic variant of X with SVQ
        """
        assert kind in ["train", "test"]
        if kind == "train":
            X = self.X_train  # (b 1 l)
        elif kind == "test":
            X = self.X_test  # (b 1 l)
        else:
            raise ValueError

        n_samples = X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test`
        zs = []
        xs_a = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x = X[s]  # (b 1 l)
            x = torch.from_numpy(x).float().to(self.device)

            # x_rec = self.stage1.forward(batch=(x, None), batch_idx=-1, return_x_rec=True).cpu().detach().numpy().astype(float)  # (b 1 l)
            # svq_temp_rng = self.config['fidelity_enhancer']['svq_temp_rng']
            # svq_temp = np.random.uniform(*svq_temp_rng)
            # tau = self.config['fidelity_enhancer']['tau']
            tau = self.fidelity_enhancer.tau.item()
            _, s_a_l = self.maskgit.encode_to_z_q(
                x, self.stage1.encoder_l, self.stage1.vq_model_l, svq_temp=tau
            )  # (b n)
            _, s_a_h = self.maskgit.encode_to_z_q(
                x, self.stage1.encoder_h, self.stage1.vq_model_h, svq_temp=tau
            )  # (b m)
            x_a_l = self.maskgit.decode_token_ind_to_timeseries(s_a_l, "lf")  # (b 1 l)
            x_a_h = self.maskgit.decode_token_ind_to_timeseries(s_a_h, "hf")  # (b 1 l)
            x_a = x_a_l + x_a_h  # (b c l)
            x_a = x_a.cpu().numpy().astype(float)
            xs_a.append(x_a)

            z_t = self._extract_feature_representations(x_a)
            zs.append(z_t)
        zs = np.concatenate(zs, axis=0)
        xs_a = np.concatenate(xs_a, axis=0)
        return zs, xs_a

    def compute_z(self, kind: str) -> np.ndarray:
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        assert kind in ["train", "test"]
        if kind == "train":
            X = self.X_train  # (b 1 l)
        elif kind == "test":
            X = self.X_test  # (b 1 l)
        else:
            raise ValueError

        n_samples = X.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_test`
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z_t = self._extract_feature_representations(X[s])
            zs.append(z_t)
        zs = np.concatenate(zs, axis=0)
        return zs

    def compute_z_gen(self, X_gen: torch.Tensor) -> np.ndarray:
        """
        It computes representation z given input x
        :param X_gen: generated X
        :return: z_test (z on X_test), z_gen (z on X_generated)
        """
        n_samples = X_gen.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors from `X_gen`
        z_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            # z_g = self.fcn(X_gen[s].float().to(self.device), return_feature_vector=True).cpu().detach().numpy()
            z_g = self._extract_feature_representations(X_gen[s].numpy().astype(float))

            z_gen.append(z_g)
        z_gen = np.concatenate(z_gen, axis=0)
        return z_gen

    def fid_score(self, z1: np.ndarray, z2: np.ndarray) -> int:
        z1, z2 = remove_outliers(z1), remove_outliers(z2)
        fid = calculate_fid(z1, z2)
        return fid

    def inception_score(self, X_gen: torch.Tensor):
        # assert self.X_test.shape[0] == X_gen.shape[0], "shape of `X_test` must be the same as that of `X_gen`."

        n_samples = self.X_test.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get the softmax distribution from `X_gen`
        p_yx_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            p_yx_g = self.fcn(X_gen[s].float().to(self.device))  # p(y|x)
            p_yx_g = torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy()

            p_yx_gen.append(p_yx_g)
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)

        IS_mean, IS_std = calculate_inception_score(p_yx_gen)
        return IS_mean, IS_std

    def stat_metrics(
        self, x_real: np.ndarray, x_gen: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        computes the statistical metrices introduced in the paper, [Ang, Yihao, et al. "Tsgbench: Time series generation benchmark." arXiv preprint arXiv:2309.03755 (2023).]
        x_real: (batch 1 length)
        x_gen: (batch 1 length)
        """
        mdd = marginal_distribution_difference(x_real, x_gen)
        acd = auto_correlation_difference(x_real, x_gen)
        sd = skewness_difference(x_real, x_gen)
        kd = kurtosis_difference(x_real, x_gen)
        return mdd, acd, sd, kd

    def log_visual_inspection(
        self,
        X1,
        X2,
        title: str,
        ylim: tuple = (-5, 5),
        n_plot_samples: int = 200,
        alpha: float = 0.1,
    ):
        b, c, l = X1.shape
        # `X_test`
        fig, axes = plt.subplots(2, c, figsize=(4 * c, 8))
        if c == 1:
            axes = axes[:, np.newaxis]
        plt.suptitle(title)
        for channel_idx in range(c):
            # X1
            sample_ind = np.random.randint(0, X1.shape[0], n_plot_samples)
            for i in sample_ind:
                # print(f"X1.shape: {X1.shape}, i: {i}, channel_idx: {channel_idx}")
                axes[0, channel_idx].plot(
                    X1[i, channel_idx, :], alpha=alpha, color="C0"
                )
            axes[0, channel_idx].set_ylim(*ylim)
            axes[0, channel_idx].set_title(f"channel idx:{channel_idx}")

            # X2
            sample_ind = np.random.randint(0, X2.shape[0], n_plot_samples)
            for i in sample_ind:
                axes[1, channel_idx].plot(
                    X2[i, channel_idx, :], alpha=alpha, color="C0"
                )
            axes[1, channel_idx].set_ylim(*ylim)

            if channel_idx == 0:
                axes[0, channel_idx].set_ylabel("X_test")
                axes[1, channel_idx].set_ylabel("X_gen")

        plt.tight_layout()
        fname = f"visual_comp_{title}.png"
        log_image(plt, fname)
        plt.close()

    def log_pca(
        self, Zs: List[np.ndarray], labels: List[str], n_plot_samples: int = 1000
    ):
        assert len(Zs) == len(labels)

        plt.figure(figsize=(4, 4))

        for Z, label in zip(Zs, labels):
            ind = np.random.choice(range(Z.shape[0]), size=n_plot_samples, replace=True)
            Z_embed = self.pca.transform(Z[ind])

            plt.scatter(Z_embed[:, 0], Z_embed[:, 1], alpha=0.1, label=label)

            xpad = (self.xmax_pca - self.xmin_pca) * 0.1
            ypad = (self.ymax_pca - self.ymin_pca) * 0.1
            plt.xlim(self.xmin_pca - xpad, self.xmax_pca + xpad)
            plt.ylim(self.ymin_pca - ypad, self.ymax_pca + ypad)

        plt.legend(loc="upper right")
        plt.tight_layout()
        fname = f"PCA_on_Z_{labels}.png"
        log_image(plt, fname)
        plt.close()

    def log_tsne(
        self, n_plot_samples: int, X_gen, z_test: np.ndarray, z_gen: np.ndarray
    ):
        X_gen = F.interpolate(
            X_gen, size=self.X_test.shape[-1], mode="linear", align_corners=True
        )
        X_gen = X_gen.cpu().numpy()

        sample_ind_test = np.random.randint(0, self.X_test.shape[0], n_plot_samples)
        sample_ind_gen = np.random.randint(0, X_gen.shape[0], n_plot_samples)

        # TNSE: data space
        X = np.concatenate(
            (self.X_test.squeeze()[sample_ind_test], X_gen.squeeze()[sample_ind_gen]),
            axis=0,
        ).squeeze()
        labels = np.array(["C0"] * len(sample_ind_test) + ["C1"] * len(sample_ind_gen))
        X_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(X)

        plt.figure(figsize=(4, 4))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, alpha=0.1)
        # plt.legend()
        plt.tight_layout()
        fname = f"TSNE_data_space.png"
        log_image(plt, fname)
        plt.close()

        # TNSE: latent space
        Z = np.concatenate(
            (z_test[sample_ind_test], z_gen[sample_ind_gen]), axis=0
        ).squeeze()
        labels = np.array(["C0"] * len(sample_ind_test) + ["C1"] * len(sample_ind_gen))
        Z_embedded = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(Z)

        plt.figure(figsize=(4, 4))
        plt.scatter(Z_embedded[:, 0], Z_embedded[:, 1], c=labels, alpha=0.1)
        # plt.legend()
        plt.tight_layout()
        fname = f"TSNE_latent_space.png"
        log_image(plt, fname)
        plt.close()
