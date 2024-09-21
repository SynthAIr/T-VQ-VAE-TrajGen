import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from timevqvae.trainers import Stage1, Stage2, Stage3
from timevqvae.utils import get_data, load_yaml_param_settings, print_dict

# Set the float32 matmul precision to 'high' to utilize Tensor Cores effectively. 'highest' default by PyTorch, 'high' is fast but less precise, 'medium' is even faster but less precise.
torch.set_float32_matmul_precision("high")

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*epoch parameter in `scheduler.step\(\)`.*",
)


def initialize_trainer(
    config: Dict[str, Any], stage: str, logger: MLFlowLogger, total_batches: int
):
    """Initialize the Lightning Trainer."""
    return Trainer(
        logger=logger,
        enable_checkpointing=False,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        max_steps=config["trainer_params"]["max_steps"][stage],
        devices=1,
        accelerator="gpu",
        val_check_interval=config["trainer_params"]["val_check_interval"][stage],
        check_val_every_n_epoch=None,
        log_every_n_steps=int(total_batches),
    )


def log_and_save_model(trainer: Trainer, model: LightningModule, save_path: str):
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.logger.log_metrics({"n_trainable_params": n_trainable_params})
    print("Saving the model...")
    trainer.save_checkpoint(save_path)


def start_mlflow_run(config: Dict[str, Any], stage: str, dataset_name: str):
    """Start MLFlow run."""
    mlflow.set_tracking_uri(config["logger"]["mlflow_uri"])
    mlflow.set_experiment(config["logger"]["experiment_name"])
    run_name = f"{dataset_name}_{stage}"
    return mlflow.start_run(run_name=run_name), run_name


def setup_mlflow_logger(config: Dict[str, Any], run_name: str, run_id: str):
    """Setup MLFlow logger."""
    return MLFlowLogger(
        experiment_name=config["logger"]["experiment_name"],
        run_name=run_name,
        tracking_uri=config["logger"]["mlflow_uri"],
        run_id=run_id,
    )


def train_stage(
    config: Dict[str, Any],
    stage: str,
    model_class: LightningModule,
    ckpt_files: Dict[str, Path],
    do_validate: bool = False,
):
    """Train a single stage of TimeVQVAE model."""
    dataset_file = config["dataset"]["file"]
    features = config["dataset"]["features"]
    batch_size = config["dataset"]["batch_sizes"][stage]
    feature_extractor_type = config["evaluation"]["feature_extractor_type"]
    train_data_loader, test_data_loader, _ = get_data(
        dataset_file, features, batch_size
    )
    total_batches = len(train_data_loader)

    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    dataset_name = os.path.basename(config["dataset"]["file"]).split(".")[0]

    model = model_class(
        input_length=input_length,
        in_channels=in_channels,
        n_classes=n_classes,
        X_train=train_data_loader.dataset.X.numpy(),
        X_test=test_data_loader.dataset.X.numpy(),
        config=config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        feature_extractor_type=feature_extractor_type,
        **ckpt_files,
    )

    run, run_name = start_mlflow_run(config, stage, dataset_name)
    run_id = run.info.run_id
    logger = setup_mlflow_logger(config, run_name, run_id)
    trainer = initialize_trainer(config, stage, logger, total_batches)

    trainer.fit(
        model,
        train_dataloaders=train_data_loader,
        val_dataloaders=test_data_loader if do_validate else None,
    )

    model_save_dir = config["logger"]["model_save_dir"]
    save_path = os.path.join(model_save_dir, dataset_name, f"{stage}.ckpt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    log_and_save_model(
        trainer,
        model,
        save_path,
    )

    mlflow.end_run()


def train(config_file: str, dataset_file: str, model_save_dir: str):
    """Main training function."""
    config = load_yaml_param_settings(config_file)
    config["dataset"]["file"] = dataset_file
    config["logger"]["model_save_dir"] = model_save_dir

    print("********************** CONFIG **********************")
    print_dict(config)

    dataset_name = os.path.basename(dataset_file).split(".")[0]

    ckpt_files_stage1 = {}
    ckpt_files_stage2 = {
        "stage1_ckpt_fname": os.path.join(
            model_save_dir,
            dataset_name,
            "stage1.ckpt",
        ),
        "fcn_ckpt_fname": os.path.join(
            model_save_dir,
            dataset_name,
            "fcn.ckpt",
        ),
    }
    ckpt_files_stage3 = {
        **ckpt_files_stage2,
        "stage2_ckpt_fname": os.path.join(
            model_save_dir,
            dataset_name,
            "stage2.ckpt",
        ),
    }

    print("********************** TRAINING: STAGE 1 **********************")
    train_stage(
        config=config, stage="stage1", model_class=Stage1, ckpt_files=ckpt_files_stage1, do_validate=False
    )

    print("********************** TRAINING: STAGE 2 **********************")
    train_stage(
        config=config, stage="stage2", model_class=Stage2, ckpt_files=ckpt_files_stage2, do_validate=False
    )

    print("********************** TRAINING: STAGE 3 **********************")
    train_stage(
        config=config, stage="stage3", model_class=Stage3, ckpt_files=ckpt_files_stage3, do_validate=False
    )


def main():
    parser = ArgumentParser(
        description="Train a Air traffic trajectory generation model using TimeVQVAE"
    )
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="configs/config.yaml"
    )
    parser.add_argument(
        "--dataset_file", type=str, required=True, help="Path to the training data file"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        help="Path to the directory where the model will be saved",
        default="saved_models",
    )
    args = parser.parse_args()
    train(args.config, args.dataset_file, args.model_save_dir)


if __name__ == "__main__":
    main()
