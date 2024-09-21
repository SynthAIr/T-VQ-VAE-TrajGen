import os
from argparse import ArgumentParser, Namespace

import lightning as L
import mlflow
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from timevqvae.models import FCNBaseline
from timevqvae.utils import get_data, load_yaml_param_settings, print_dict

torch.set_float32_matmul_precision("high")

def detach_the_unnecessary(loss_hist: dict):
    """
    apply `.detach()` on Tensors that do not need back-prop computation.
    :return:
    """
    for k in loss_hist.keys():
        if k not in ["loss"]:
            try:
                loss_hist[k] = loss_hist[k].detach()
            except AttributeError:
                pass


def compute_avg_outs(outs: dict):
    mean_outs = {}
    for k in outs[0].keys():
        mean_outs.setdefault(k, 0.0)
        for i in range(len(outs)):
            mean_outs[k] += outs[i][k]
        mean_outs[k] /= len(outs)
    return mean_outs


def get_log_items_epoch(kind: str, current_epoch: int, mean_outs: dict):
    log_items_ = {f"{kind}/{k}": v for k, v in mean_outs.items()}
    log_items = {"epoch": current_epoch}
    log_items = dict(log_items.items() | log_items_.items())
    return log_items


def get_log_items_global_step(kind: str, global_step: int, out: dict):
    log_items_ = {f"{kind}/{k}": v for k, v in out.items()}
    log_items = {"global_step": global_step}
    log_items = dict(log_items.items() | log_items_.items())
    return log_items


class ExpBase(L.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError


#     def on_train_epoch_end(self):
#         outs = self.trainer.callback_metrics
#         mean_outs = self.compute_avg_outs(outs)
#         log_items = self.get_log_items_epoch('train', self.current_epoch, mean_outs)
#         mlflow.log_metrics(log_items, step=self.current_epoch)

#     def on_train_batch_end(self, outputs, batch, batch_idx):
#         log_items = self.get_log_items_global_step('train', self.global_step, outputs)
#         mlflow.log_metrics(log_items, step=self.global_step)

#     def on_validation_epoch_end(self):
#         outs = self.trainer.callback_metrics
#         mean_outs = self.compute_avg_outs(outs)
#         log_items = self.get_log_items_epoch('val', self.current_epoch, mean_outs)
#         mlflow.log_metrics(log_items, step=self.current_epoch)

#     def on_validation_batch_end(self, outputs, batch, batch_idx):
#         log_items = self.get_log_items_global_step('val', self.global_step, outputs)
#         mlflow.log_metrics(log_items, step=self.global_step)

#     def configure_optimizers(self):
#         raise NotImplementedError

#     def on_test_epoch_end(self):
#         outs = self.trainer.callback_metrics
#         mean_outs = self.compute_avg_outs(outs)
#         log_items = self.get_log_items_epoch('test', self.current_epoch, mean_outs)
#         mlflow.log_metrics(log_items, step=self.current_epoch)

#     def on_test_batch_end(self, outputs, batch, batch_idx):
#         log_items = self.get_log_items_global_step('test', self.global_step, outputs)
#         mlflow.log_metrics(log_items, step=self.global_step)


class ExpFCN(ExpBase):
    def __init__(
        self,
        config: dict,
        n_train_samples: int,
        n_classes: int,
    ):
        super().__init__()
        self.config = config
        self.T_max = config["trainer_params"]["max_epochs"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_size"]) + 1
        )
        in_channels = config["dataset"]["in_channels"]

        self.fcn = FCNBaseline(in_channels, n_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        acc = accuracy_score(
            y.flatten().detach().cpu().numpy(),
            yhat.argmax(dim=-1).flatten().cpu().detach().numpy(),
        )
        loss_hist = {"loss": loss, "acc": acc}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(
            y.flatten().detach().cpu().numpy(),
            yhat.argmax(dim=-1).flatten().cpu().detach().numpy(),
        )
        loss_hist = {"loss": loss, "acc": acc}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.fcn.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
            ],
            weight_decay=self.config["exp_params"]["weight_decay"],
        )
        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(
            y.flatten().detach().cpu().numpy(),
            yhat.argmax(dim=-1).flatten().cpu().detach().numpy(),
        )
        loss_hist = {"loss": loss, "acc": acc}

        detach_the_unnecessary(loss_hist)
        return loss_hist


def run(args: Namespace):

    config = load_yaml_param_settings(args.config)
    config["dataset"]["file"] = args.dataset_file
    config["logger"]["model_save_dir"] = args.model_save_dir

    print("********************** CONFIG **********************")
    print_dict(config)

    # data pipeline

    dataset_file = config["dataset"]["file"]
    features = config["dataset"]["features"]
    batch_size = config["dataset"]["batch_size"]
    train_data_loader, test_data_loader, _ = get_data(
        dataset_file, features, batch_size
    )
    total_batches = len(train_data_loader)

    # fit
    train_exp = ExpFCN(
        config,
        len(train_data_loader.dataset),
        len(np.unique(train_data_loader.dataset.Y)),
    )

    tracking_uri = config["logger"]["mlflow_uri"]
    experiment_name = config["logger"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    dataset_name = os.path.basename(config["dataset"]["file"]).split(".")[0]
    run_name = f"{dataset_name}_stage1"

    with mlflow.start_run(run_name=run_name) as run:

        run_id = run.info.run_id

        l_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            run_id=run_id,
        )

        trainer = Trainer(
            logger=l_logger,
            enable_checkpointing=False,
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
            max_steps=config["trainer_params"]["max_epochs"],
            devices=config["trainer_params"]["gpus"],
            accelerator="gpu",
            log_every_n_steps=int(total_batches),
        )

        trainer.fit(
            train_exp,
            train_dataloaders=train_data_loader,
            val_dataloaders=test_data_loader,
        )

        # test
        trainer.test(train_exp, test_data_loader)

        print("saving the model...")
        model_save_dir = config["logger"]["model_save_dir"]
        save_path = os.path.join(model_save_dir, dataset_name, "fcn.ckpt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(train_exp.fcn.state_dict(), save_path)


def main():

    parser = ArgumentParser(description="Train a supervised FCN model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
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
    run(args)


if __name__ == "__main__":
    main()
