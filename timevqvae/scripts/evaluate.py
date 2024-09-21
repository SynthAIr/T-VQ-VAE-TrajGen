import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from timevqvae.generation import TrainedModelSampler
from timevqvae.utils import (get_data, get_root_dir, load_yaml_param_settings,
                             log_image, set_seed, str2bool)


def evaluate(config, dataset_file, model_save_dir, use_fidelity_enhancer):
    config = load_yaml_param_settings(config)
    config["dataset"]["file"] = dataset_file
    config["logger"]["model_save_dir"] = model_save_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the numpy matrix of the test samples
    features = config["dataset"]["features"]
    batch_size = config["evaluation"]["batch_size"]

    train_data_loader, test_data_loader, _ = get_data(
        dataset_file, features, batch_size
    )
    X_train, X_test, Y_train, Y_test = (
        train_data_loader.dataset.X.numpy(),
        test_data_loader.dataset.X.numpy(),
        train_data_loader.dataset.Y,
        test_data_loader.dataset.Y,
    )
    n_classes = len(np.unique(Y_train))
    _, in_channels, input_length = X_train.shape

    seed = 42
    set_seed(seed)

    dataset_name = os.path.basename(config["dataset"]["file"]).split(".")[0]
    stage1_ckpt_fname = os.path.join(model_save_dir, dataset_name, "stage1.ckpt")
    stage2_ckpt_fname = os.path.join(model_save_dir, dataset_name, "stage2.ckpt")
    stage3_ckpt_fname = os.path.join(model_save_dir, dataset_name, "stage3.ckpt")
    fcn_ckpt_fname = os.path.join(model_save_dir, dataset_name, "fcn.ckpt")

    tracking_uri = config["logger"]["mlflow_uri"]
    experiment_name = config["logger"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_name = f"{dataset_name}_evaluation"

    with mlflow.start_run(run_name=run_name) as run:

        run_id = run.info.run_id

        # unconditional sampling
        print("evaluating...")

        evaluation = TrainedModelSampler(
            stage1_ckpt_fname=stage1_ckpt_fname,
            stage2_ckpt_fname=stage2_ckpt_fname,
            stage3_ckpt_fname=stage3_ckpt_fname,
            fcn_ckpt_fname=fcn_ckpt_fname,
            input_length=input_length,
            in_channels=in_channels,
            n_classes=n_classes,
            batch_size=batch_size,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            device=device,
            config=config,
            use_fidelity_enhancer=use_fidelity_enhancer,
            feature_extractor_type=config["evaluation"]["feature_extractor_type"],
            do_evaluate=True,
        ).to(device)

        min_num_gen_samples = config["evaluation"][
            "min_num_gen_samples"
        ]  # large enough to capture the distribution
        (_, _, xhat), xhat_R = evaluation.sample(
            max(evaluation.X_test.shape[0], min_num_gen_samples), "unconditional"
        )
        z_train = evaluation.z_train
        z_test = evaluation.z_test
        z_rec_train = evaluation.compute_z_rec("train")
        z_rec_test = evaluation.compute_z_rec("test")
        zhat = evaluation.compute_z_gen(xhat)

        print("evaluation for unconditional sampling...")
        # wandb.log({'FID': evaluation.fid_score(z_test, zhat)})
        mlflow.log_metric("FID", evaluation.fid_score(z_test, zhat))
        IS_mean, IS_std = evaluation.inception_score(xhat)
        # wandb.log({'IS_mean': IS_mean, 'IS_std': IS_std})
        mlflow.log_metric("IS_mean", IS_mean)
        mlflow.log_metric("IS_std", IS_std)

        evaluation.log_visual_inspection(evaluation.X_train, xhat, "X_train vs Xhat")
        evaluation.log_visual_inspection(evaluation.X_test, xhat, "X_test vs Xhat")
        evaluation.log_visual_inspection(
            evaluation.X_train, evaluation.X_test, "X_train vs X_test"
        )

        evaluation.log_pca(
            [
                z_train,
            ],
            [
                "Z_train",
            ],
        )
        evaluation.log_pca(
            [
                z_test,
            ],
            [
                "Z_test",
            ],
        )
        evaluation.log_pca(
            [
                zhat,
            ],
            [
                "Zhat",
            ],
        )

        evaluation.log_pca([z_train, zhat], ["Z_train", "Zhat"])
        evaluation.log_pca([z_test, zhat], ["Z_test", "Zhat"])
        evaluation.log_pca([z_train, z_test], ["Z_train", "Z_test"])

        evaluation.log_pca([z_train, z_rec_train], ["Z_train", "Z_rec_train"])
        evaluation.log_pca([z_test, z_rec_test], ["Z_test", "Z_rec_test"])
        mdd, acd, sd, kd = evaluation.stat_metrics(evaluation.X_test, xhat)
        # wandb.log({'MDD':mdd, 'ACD':acd, 'SD':sd, 'KD':kd})
        mlflow.log_metric("MDD", mdd)
        mlflow.log_metric("ACD", acd)
        mlflow.log_metric("SD", sd)
        mlflow.log_metric("KD", kd)

        # if use_fidelity_enhancer:
        if use_fidelity_enhancer:
            z_svq_train, x_prime_train = evaluation.compute_z_svq("train")
            z_svq_test, x_prime_test = evaluation.compute_z_svq("test")
            zhat_R = evaluation.compute_z_gen(xhat_R)

            evaluation.log_pca(
                [
                    z_svq_train,
                ],
                [
                    "Z_svq_train",
                ],
            )
            evaluation.log_pca(
                [
                    z_svq_test,
                ],
                [
                    "Z_svq_test",
                ],
            )
            evaluation.log_visual_inspection(
                x_prime_train, x_prime_test, "X_prime_train & X_prime_test"
            )
            evaluation.log_pca([z_train, z_svq_train], ["Z_train", "Z_svq_train"])
            evaluation.log_pca([z_test, z_svq_test], ["Z_test", "Z_svq_test"])

            IS_mean, IS_std = evaluation.inception_score(xhat_R)
            # wandb.log({'FID w/ FE': evaluation.fid_score(z_test, zhat_R),
            #         'IS_mean w/ FE': IS_mean,
            #         'IS_std w/ FE': IS_std})

            mlflow.log_metric("FID with FE", evaluation.fid_score(z_test, zhat_R))
            mlflow.log_metric("IS_mean with FE", IS_mean)
            mlflow.log_metric("IS_std wwith FE", IS_std)

            evaluation.log_visual_inspection(
                evaluation.X_train, xhat_R, "X_train vs Xhat_R"
            )
            evaluation.log_visual_inspection(
                evaluation.X_test, xhat_R, "X_test vs Xhat_R"
            )
            evaluation.log_visual_inspection(
                xhat[[0]], xhat_R[[0]], "xhat vs xhat_R", alpha=1.0, n_plot_samples=1
            )  # visaulize a single pair
            evaluation.log_pca(
                [
                    zhat_R,
                ],
                [
                    "Zhat_R",
                ],
            )
            evaluation.log_pca([z_train, zhat_R], ["Z_train", "Zhat_R"])
            evaluation.log_pca([z_test, zhat_R], ["Z_test", "Zhat_R"])

            mdd, acd, sd, kd = evaluation.stat_metrics(evaluation.X_test, xhat_R)
            # wandb.log({'MDD with FE':mdd, 'ACD with FE':acd, 'SD with FE':sd, 'KD with FE':kd})
            mlflow.log_metric("MDD with FE", mdd)
            mlflow.log_metric("ACD with FE", acd)
            mlflow.log_metric("SD with FE", sd)
            mlflow.log_metric("KD with FE", kd)

        # class-conditional sampling
        print("evaluation for class-conditional sampling...")
        n_plot_samples_per_class = 100  # 200
        alpha = 0.1
        ylim = (-5, 5)
        n_rows = int(np.ceil(np.sqrt(n_classes)))
        fig1, axes1 = plt.subplots(n_rows, n_rows, figsize=(4 * n_rows, 2 * n_rows))
        fig2, axes2 = plt.subplots(n_rows, n_rows, figsize=(4 * n_rows, 2 * n_rows))
        fig3, axes3 = plt.subplots(n_rows, n_rows, figsize=(4 * n_rows, 2 * n_rows))
        fig1.suptitle("X_test_c")
        fig2.suptitle(f"Xhat_c (cfg_scale-{config['MaskGIT']['cfg_scale']})")
        fig3.suptitle(f"Xhat_R_c (cfg_scale-{config['MaskGIT']['cfg_scale']})")
        axes1 = axes1.flatten()
        axes2 = axes2.flatten()
        axes3 = axes3.flatten()
        for cls_idx in range(n_classes):
            (_, _, xhat_c), xhat_c_R = evaluation.sample(
                n_plot_samples_per_class, kind="conditional", class_index=cls_idx
            )
            cls_sample_ind = evaluation.Y_test[:, 0] == cls_idx  # (b,)

            X_test_c = evaluation.X_test[cls_sample_ind]  # (b' 1 l)
            sample_ind = np.random.randint(
                0, X_test_c.shape[0], n_plot_samples_per_class
            )
            axes1[cls_idx].plot(X_test_c[sample_ind, 0, :].T, alpha=alpha, color="C0")
            axes1[cls_idx].set_title(f"cls_idx:{cls_idx}")
            axes1[cls_idx].set_ylim(*ylim)

            sample_ind = np.random.randint(0, xhat_c.shape[0], n_plot_samples_per_class)
            axes2[cls_idx].plot(xhat_c[sample_ind, 0, :].T, alpha=alpha, color="C0")
            axes2[cls_idx].set_title(f"cls_idx:{cls_idx}")
            axes2[cls_idx].set_ylim(*ylim)

            # if use_fidelity_enhancer:
            if use_fidelity_enhancer:
                sample_ind = np.random.randint(
                    0, xhat_c_R.shape[0], n_plot_samples_per_class
                )
                axes3[cls_idx].plot(
                    xhat_c_R[sample_ind, 0, :].T, alpha=alpha, color="C0"
                )
                axes3[cls_idx].set_title(f"cls_idx:{cls_idx}")
                axes3[cls_idx].set_ylim(*ylim)

        fig1.tight_layout()
        fig2.tight_layout()
        # wandb.log({"X_test_c": wandb.Image(fig1)})
        fname = "X_test_c.png"
        log_image(fig1, fname)
        # wandb.log({f"Xhat_c": wandb.Image(fig2)})
        fname = "Xhat_c.png"
        log_image(fig2, fname)

        # if use_fidelity_enhancer:
        if use_fidelity_enhancer:
            fig3.tight_layout()
            # wandb.log({f"Xhat_R_c": wandb.Image(fig3)})
            fname = "Xhat_R_c.png"
            log_image(fig3, fname)

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def main():

    parser = ArgumentParser(
        description="Evaluate the TimeVQVAE model for generating synthetic trajectories."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config data  file.",
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--dataset_file", type=str, required=True, help="Path to the training data file"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        help="Path to the directory where the models are saved",
        default="saved_models",
    )
    parser.add_argument(
        "--use_fidelity_enhancer",
        type=str2bool,
        help="Whether to use the fidelity enhancer",
        default=True,
    )
    args = parser.parse_args()
    evaluate(
        args.config, args.dataset_file, args.model_save_dir, args.use_fidelity_enhancer
    )

    # clean memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
