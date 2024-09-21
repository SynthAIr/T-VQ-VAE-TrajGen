"""
FID, IS
"""

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eval_utils import (calculate_fid,
                                       calculate_inception_score,
                                       load_pretrained_FCN)
from .rocket_functions import apply_kernels, generate_kernels
from .stat_metrics import (auto_correlation_difference,
                                         kurtosis_difference,
                                         marginal_distribution_difference,
                                         skewness_difference)
from timevqvae.utils import (conditional_sample, freeze, remove_outliers,
                             unconditional_sample)


@torch.no_grad()
def sample(
    batch_size: int,
    maskgit,
    device,
    n_samples: int,
    kind: str,
    class_index: Union[None, int],
):
    assert kind in ["unconditional", "conditional"]

    # sampling
    if kind == "unconditional":
        x_new_l, x_new_h, x_new = unconditional_sample(
            maskgit, n_samples, device, batch_size=batch_size
        )  # (b c l); b=n_samples, c=1 (univariate)
    elif kind == "conditional":
        x_new_l, x_new_h, x_new = conditional_sample(
            maskgit, n_samples, device, class_index, batch_size
        )  # (b c l); b=n_samples, c=1 (univariate)
    else:
        raise ValueError

    return x_new_l, x_new_h, x_new


class Metrics(nn.Module):
    """
    - FID
    - IS
    - visual inspection
    - PCA
    - t-SNE
    """

    def __init__(
        self,
        fcn_ckpt_fname,
        input_length: int,
        in_channels: int,
        n_classes: int,
        batch_size: int,
        X_train: np.ndarray,
        X_test: np.ndarray,
        device: torch.device,
        feature_extractor_type: str,
        rocket_num_kernels: int = 1000,
    ):
        super().__init__()
        self.feature_extractor_type = feature_extractor_type
        self.batch_size = batch_size
        self.device = device

        # load the numpy matrix of the test samples
        self.X_train = X_train  # (b 1 l)
        self.X_test = X_test  # (b 1 l)
        self.n_classes = n_classes

        # load a model
        if self.feature_extractor_type == "supervised_fcn":
            self.fcn = load_pretrained_FCN(
                ckpt_fname=fcn_ckpt_fname, in_channels=in_channels, n_classes=n_classes
            ).to(self.device)
            freeze(self.fcn)
            self.fcn.eval()
        elif self.feature_extractor_type == "rocket":
            input_length = self.X_train.shape[-1]
            self.rocket_kernels = generate_kernels(
                input_length, num_kernels=rocket_num_kernels
            )
        else:
            raise ValueError

        # compute z_train, z_test
        self.z_train = self.compute_z(self.X_train)  # (b d)
        self.z_test = self.compute_z(self.X_test)  # (b d)

    @torch.no_grad()
    def sample(
        self, maskgit, device, n_samples: int, kind: str, class_index: Union[None, int]
    ):
        return sample(self.batch_size, maskgit, device, n_samples, kind, class_index)

    def extract_feature_representations(self, x: np.ndarray):
        """
        x: (b 1 l)
        """
        if self.feature_extractor_type == "supervised_fcn":
            device = next(self.fcn.parameters()).device
            z = (
                self.fcn(
                    torch.from_numpy(x).float().to(device), return_feature_vector=True
                )
                .cpu()
                .detach()
                .numpy()
            )  # (b d)
        elif self.feature_extractor_type == "rocket":
            x = x[:, 0, :].astype(float)  # (b l)
            z = apply_kernels(x, self.rocket_kernels)  # (b d)
            z = F.normalize(torch.from_numpy(z).float(), p=2, dim=-1).numpy()
        else:
            raise ValueError
        return z

    def compute_z_stat(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z = self.extract_feature_representations(x[s])
            zs.append(z)
        zs = np.concatenate(zs, axis=0)
        z_mu, z_std = (
            np.mean(zs, axis=0)[None, :],
            np.std(zs, axis=0)[None, :],
        )  # (1 d), (1 d)
        return z_mu, z_std

    def compute_z(self, x: np.ndarray) -> np.ndarray:
        n_samples = x.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get feature vectors
        zs = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            z = self.extract_feature_representations(x[s])
            zs.append(z)
        zs = np.concatenate(zs, axis=0)
        return zs

    def z_gen_fn(self, x_gen: np.ndarray):
        z_gen = self.compute_z(x_gen)
        return z_gen

    # def fid_score(self, x_gen: np.ndarray):
    #     z_gen = self.z_gen_fn(x_gen)
    #     z_gen = remove_outliers(z_gen)

    #     fid_train_gen = calculate_fid(self.z_train, z_gen)
    #     fid_test_gen = calculate_fid(self.z_test, z_gen)
    #     return fid_train_gen, fid_test_gen

    def fid_score(self, z1: np.ndarray, z2: np.ndarray) -> int:
        z1, z2 = remove_outliers(z1), remove_outliers(z2)
        fid = calculate_fid(z1, z2)
        return fid

    def inception_score(self, x_gen: np.ndarray):
        device = next(self.fcn.parameters()).device
        n_samples = x_gen.shape[0]
        n_iters = n_samples // self.batch_size
        if n_samples % self.batch_size > 0:
            n_iters += 1

        # get the softmax distribution from `x_gen`
        p_yx_gen = []
        for i in range(n_iters):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            p_yx_g = self.fcn(torch.from_numpy(x_gen[s]).float().to(device))  # p(y|x)
            p_yx_g = torch.softmax(p_yx_g, dim=-1).cpu().detach().numpy()

            p_yx_gen.append(p_yx_g)
        p_yx_gen = np.concatenate(p_yx_gen, axis=0)

        IS_mean, IS_std = calculate_inception_score(p_yx_gen, n_split=5)
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
