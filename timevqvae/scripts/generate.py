import os
from argparse import ArgumentParser
from collections import Counter

import numpy as np
import pandas as pd
import torch
from traffic.core import Traffic

from timevqvae.generation import TrainedModelSampler
from timevqvae.utils import get_data, get_root_dir, load_yaml_param_settings


def post_processed_generated_trajectories(x_gen, y_gen, scaler, features):
    x_gen = x_gen.detach().transpose(1, 2).reshape(x_gen.shape[0], -1)
    x_gen = x_gen.cpu().numpy()
    x_gen = scaler.inverse_transform(x_gen)
    # Neural nets might not predict exactly timedelta = 0 for the first observation
    x_gen[:, 3] = 0  # Assuming the fourth column corresponds to timedelta in your dataset
    n_samples = x_gen.shape[0]
    x_gen = x_gen.reshape(n_samples, -1, len(features))
    n_obs = x_gen.shape[1]
    df = pd.DataFrame(
        {feature: x_gen[:, :, i].ravel() for i, feature in enumerate(features)}
    )
    # Enriches DataFrame with flight_id, callsign, and icao24 columns
    ids = np.array([[f"TRAJ_{sample}"] * n_obs for sample in range(n_samples)]).ravel()
    df = df.assign(flight_id=ids, callsign=ids, icao24=ids)

    labels = y_gen.cpu().numpy()
    cluster = np.array([[labels[i]] * n_obs for i in range(n_samples)]).ravel()
    df = df.assign(cluster=cluster)

    # Clip altitude values below zero
    df.loc[df.altitude < 0, "altitude"] = 0

    if "timedelta" in df.columns:
        base_ts = pd.Timestamp.today(tz="UTC").round(freq="s")
        df = df.assign(timestamp=pd.to_timedelta(df.timedelta, unit="s") + base_ts)

    return Traffic(df)


def generate_synthetic_data(config, dataset_file, model_save_dir, synthetic_save_dir, use_fidelity_enhancer):
    config = load_yaml_param_settings(config)
    config["dataset"]["file"] = dataset_file
    config["logger"]["model_save_dir"] = model_save_dir

    features = config["dataset"]["features"]
    batch_size = config["evaluation"]["batch_size"]
    train_data_loader, test_data_loader, scaler = get_data(dataset_file, features, batch_size)

    X_train, X_test, Y_train, Y_test = (
        train_data_loader.dataset.X.numpy(),
        test_data_loader.dataset.X.numpy(),
        train_data_loader.dataset.Y,
        test_data_loader.dataset.Y,
    )

    dataset_name = os.path.basename(dataset_file).split(".")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train_samples_per_class = dict(Counter(train_data_loader.dataset.Y.flatten()))
    n_test_samples_per_class = dict(Counter(test_data_loader.dataset.Y.flatten()))
    all_samples_per_class = dict(Counter(n_train_samples_per_class) + Counter(n_test_samples_per_class))

    print("n_train_samples_per_class:", n_train_samples_per_class)

    stage1_ckpt_fname = os.path.join(model_save_dir, dataset_name, "stage1.ckpt")
    stage2_ckpt_fname = os.path.join(model_save_dir, dataset_name, "stage2.ckpt")
    stage3_ckpt_fname = os.path.join(model_save_dir, dataset_name, "stage3.ckpt")
    fcn_ckpt_fname = os.path.join(model_save_dir, dataset_name, "fcn.ckpt")

    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape

    sampler = TrainedModelSampler(
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
        feature_extractor_type=config["evaluation"]["feature_extractor_type"],
        do_evaluate=False,
    ).to(device)

    X_gen, Y_gen = [], []
    if use_fidelity_enhancer:
        for class_idx, n_samples in all_samples_per_class.items():
            print(f"sampling synthetic data | class_idx: {class_idx}...")
            _, x_gen = sampler.sample(
                n_samples=n_samples, kind="conditional", class_index=class_idx
            )
            X_gen.append(x_gen)
            Y_gen.append(torch.Tensor([class_idx] * n_samples))
    else:
        for class_idx, n_samples in all_samples_per_class.items():
            print(f"sampling synthetic data | class_idx: {class_idx}...")
            (_, _, x_gen), _ = sampler.sample(
                n_samples=n_samples, kind="conditional", class_index=class_idx
            )
            X_gen.append(x_gen)
            Y_gen.append(torch.Tensor([class_idx] * n_samples))
    
    X_gen = torch.cat(X_gen).float()
    Y_gen = torch.cat(Y_gen)[:, None].long()

    print("X_gen.shape:", X_gen.shape)
    print("Y_gen.shape:", Y_gen.shape)

    traffic = post_processed_generated_trajectories(
        X_gen, Y_gen, scaler, config["dataset"]["features"]
    )

    os.makedirs(synthetic_save_dir, exist_ok=True)
    save_fname = os.path.basename(config["dataset"]["file"])
    traffic.to_pickle(os.path.join(synthetic_save_dir, save_fname))


def main():
    parser = ArgumentParser(description="Generate synthetic data with and without fidelity enhancer.")
    parser.add_argument(
        "--config", type=str, help="Path to the config data file.", default="configs/config.yaml"
    )
    parser.add_argument(
        "--dataset_file", type=str, required=True, help="Path to the training data file"
    )
    parser.add_argument(
        "--model_save_dir", type=str, help="Path to the directory where the models are saved", default="saved_models"
    )
    parser.add_argument(
        "--synthetic_save_dir", type=str, help="Path to the directory where the generated samples will be saved", default="./data/synthetic"
    )
    parser.add_argument(
        "--synthetic_fidelity_dir", type=str, help="Path to the directory where the fidelity-enhanced samples will be saved", default="./data/synthetic_fidelity"
    )
    args = parser.parse_args()

    # Generate normal synthetic data
    generate_synthetic_data(
        args.config, args.dataset_file, args.model_save_dir, args.synthetic_save_dir, use_fidelity_enhancer=False
    )

    # Generate synthetic data with fidelity enhancer
    generate_synthetic_data(
        args.config, args.dataset_file, args.model_save_dir, args.synthetic_fidelity_dir, use_fidelity_enhancer=True
    )


if __name__ == "__main__":
    main()
