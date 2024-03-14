import os
from datetime import datetime
from itertools import count
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from confit import Cli
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.functional import f1_score, precision

from oeciml.torchmodules.data.dataset import TextDataset, collate_fn
from oeciml.torchmodules.helpers.misc import save_model
from oeciml.torchmodules.loss import PytorchLoss
from oeciml.torchmodules.model import (
    QualifierModelv2,
    ToyModel,
    tokenizer_from_pretrained,
)
from oeciml.torchmodules.optimizer import CyclicalLinearSchedule, ScheduledOptimizer

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="script")
def main(
    path_moving_loss: str,
    path_loss_values: str,
    num_accumulation_steps: int = 1,  # Gradient accumulation
    batch_size: int = 64,  # effective batch size
    total_steps: Optional[int] = None,
    n_epochs: Optional[int] = None,
    dropout_classification_head: float = 0,
    log_each_n_steps: int = 10,
    log_each_device: bool = False,
    path_save_model: Optional[str] = None,
    optimizer_params: Dict[str, Any] = dict(
        epochs_per_cycle=4,
        head=dict(
            max_value=5e-4,
            min_value=1e-6,
        ),
        transformer=dict(
            max_value=5e-5,
            min_value=1e-8,
        ),
    ),
    num_classes: int = 3,
    loss_fn: _Loss = PytorchLoss("CrossEntropyLoss", reduction="none"),
    experience_name: str = "O2UToyModel",
    path_embedding: str = "/data/scratch/cse/camembert-EDS",
    path_tensorboard: str = "/export/home/cse/tensorboard_data/",
    path_tokenizer: str = "/data/scratch/cse/camembert-EDS/",
    path_train_dataset: str = "/data/scratch/cse/oeci-ml/data/config_all_cancer/dataset_debug",
    path_dev_dataset: Optional[
        str
    ] = "/data/scratch/cse/oeci-ml/data/config_base/dataset_dev/",
    binarize_validation: bool = True,
    debug: bool = False,
    seed: Optional[int] = 54,
):
    if seed is None:
        seed = np.random.randint(0, 1000)
    set_seed(seed)
    assert bool(total_steps) != bool(n_epochs)

    # Set env variables
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Tracker (tensorboard)
    path_tensorboard = os.path.join(
        path_tensorboard,
        experience_name,
        datetime.today().strftime("%Y_%m_%d_T_%H_%M_%S"),
    )

    # Accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=num_accumulation_steps,
        log_with="tensorboard",
        project_dir=path_tensorboard,
        kwargs_handlers=[kwargs],
    )

    device = accelerator.device
    accelerator.print(f"Device : {device} num_processes: {accelerator.num_processes}")
    accelerator.print("Project dir\n", accelerator.project_dir)

    # Tokenizer
    tokenizer = tokenizer_from_pretrained(path=path_tokenizer)

    # Data
    set_seed(seed)
    dataset_train = TextDataset(path_train_dataset, tokenizer, split="train")

    set_seed(seed)
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size // (num_accumulation_steps * accelerator.num_processes),
        shuffle=True,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    set_seed(seed)

    if path_dev_dataset is not None:
        dataset_valid = TextDataset(path_dev_dataset, tokenizer, split="val")
        val_dataloader = DataLoader(
            dataset_valid,
            batch_size=batch_size
            // (num_accumulation_steps * accelerator.num_processes),
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        val_dataloader = None

    # Model
    set_seed(seed)
    if debug:
        model = ToyModel(num_classes=num_classes)
    else:
        model = QualifierModelv2(
            path_embedding=path_embedding,
            num_classes=num_classes,
            dropout_classification_head=dropout_classification_head,
            **dict(tokenizer=tokenizer),
        )

    # Prepare for ressources
    (train_dataloader, val_dataloader) = accelerator.prepare(
        train_dataloader, val_dataloader
    )

    # Number of steps and epochs
    steps_per_epoch = len(train_dataloader)
    if total_steps is None:
        total_steps = int(steps_per_epoch * n_epochs)
    if n_epochs is None:
        n_epochs = total_steps // steps_per_epoch

    accelerator.print(
        f"""
Number of steps {total_steps}
Number of epochs {n_epochs}
Steps per epoch {steps_per_epoch}
Batch size {batch_size}
Train dataset length {len(dataset_train)}
Train datasloader x batch_size {len(train_dataloader) *  batch_size}
        """
    )

    # Optimizer
    optimizer_params["total_steps"] = total_steps

    # # update weights only for the classification head and not for the transformer part
    # for param in model.transformer.parameters():
    #     param.requires_grad = False

    # for param in model.transformer.encoder.layer[-2:].parameters():
    #     param.requires_grad = True

    # for param in model.transformer.pooler.parameters():
    #     param.requires_grad = True

    # apply the cyclical learning rate
    optimizer = ScheduledOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": model.classification_head.parameters(),
                    "lr": optimizer_params["head"]["max_value"],
                    "schedules": CyclicalLinearSchedule(
                        path="lr",
                        max_value=optimizer_params["head"]["max_value"],
                        min_value=optimizer_params["head"]["min_value"],
                        epochs_per_cycle=optimizer_params["epochs_per_cycle"],
                        steps_per_epoch=steps_per_epoch,
                    ),
                    # "momentum": 0.9,  # following o2u paper
                    # "weight_decay": 0,  # o2u paper: 5e-4
                },
                {
                    "params": model.transformer.parameters(),
                    "lr": optimizer_params["transformer"]["max_value"],
                    "schedules": CyclicalLinearSchedule(
                        path="lr",
                        max_value=optimizer_params["transformer"]["max_value"],
                        min_value=optimizer_params["transformer"]["min_value"],
                        epochs_per_cycle=optimizer_params["epochs_per_cycle"],
                        steps_per_epoch=steps_per_epoch,
                    ),
                    # "momentum": 0.9,  # following o2u paper
                    # "weight_decay": 0,  # o2u paper: 5e-4
                },
            ]
        )
    )
    optimizer, model = accelerator.prepare(optimizer, model)

    # Metrics
    valid_metrics = {
        "f1": F1Score(task="binary"),
        "precision": Precision(task="binary"),
        "recall": Recall(task="binary"),
    }
    for metric_name, metric in valid_metrics.items():
        valid_metrics[metric_name] = metric.to(accelerator.device)  # todo prepare ?

    # Tracker (Tensorboard and hparams)
    hps = {
        "num_iterations": total_steps,
        "batch_size": batch_size,
        "loss": loss_fn,
        "num_classes": num_classes,
        "path_dataset": path_train_dataset,
        "experience_name": experience_name,
        "path_tokenizer": path_tokenizer,
        "seed": seed,
        "debug": debug,
    }
    hps["optimizer_params"] = optimizer_params
    accelerator.init_trackers("")
    if accelerator.is_main_process:
        with open(os.path.join(path_tensorboard, "hparams.yml"), "w") as outfile:
            try:
                yaml.dump(hps, outfile)
            except yaml.representer.RepresenterError:
                raise

    # O2U
    ntrain = len(dataset_train)
    sample_loss = torch.zeros(ntrain, dtype=torch.float32)
    moving_loss = torch.zeros(ntrain, dtype=torch.float32)
    loss_values = torch.zeros([ntrain, n_epochs], dtype=torch.float32)

    # Train loop
    set_seed(seed)
    model.train()
    train_iterator = (batch for epoch in count() for batch in train_dataloader)
    accelerator.print("## Training ##")
    epoch = -1
    for step_idx in range(total_steps):
        # Epoch number
        if step_idx % steps_per_epoch == 0:
            epoch += 1
            accelerator.print(f"### Starting epoch {epoch} ###")

        # Step
        accu_loss = torch.tensor([0.0]).to(accelerator.device)
        for accu_step_idx in range(num_accumulation_steps):
            with accelerator.accumulate(model):
                batch = next(train_iterator)

                # Forward
                logits, preds, probs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["span_start"],
                    batch["span_end"],
                )

                # Loss
                loss = loss_fn(logits, batch["label"].long())

                # Loss (gradient accumulation)
                accu_loss += loss.detach().sum()

                # Backpropagation
                accelerator.backward(loss.sum())

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

            # O2U
            indices, loss = accelerator.gather_for_metrics(
                (batch["id"].flatten(), loss.detach())
            )
            loss = loss.cpu()
            indices = indices.view(-1).cpu().numpy()

            sample_loss[indices] = loss
            loss_values[indices, epoch] = loss

        # Loss of all devices
        loss_all_devices = accelerator.gather(accu_loss).sum()

        # Log (training steps)
        if step_idx % log_each_n_steps == 0:
            # Log to output
            accelerator.print(
                f"loss: {loss_all_devices.item():>7f}   [{step_idx:>5d}/{total_steps:>5d}]"
            )

            # Get lr
            lr_classification_head = optimizer.optimizer.param_groups[0]["lr"]

            # Log to tensorboard
            accelerator.log(
                {
                    "train/loss": loss_all_devices.item(),
                    "lr_classification_head": lr_classification_head,
                },
                step=step_idx,
            )

            # Log each device
            if log_each_device:
                accu_loss = accelerator.gather(accu_loss)
                for i, loss_gpu in enumerate(accu_loss):
                    accelerator.log(
                        {f"train/loss/device={str(i)}": loss_gpu.item()}, step=step_idx
                    )

        # ON EPOCH END
        # applying O2U scheme
        if (step_idx + 1) % steps_per_epoch == 0:
            accelerator.wait_for_everyone()
            accelerator.print("### EPOCH END ###")
            if accelerator.is_main_process:
                epoch_loss_mean = sample_loss.mean().item()
                moving_loss += sample_loss - epoch_loss_mean
                sample_loss = torch.zeros(ntrain, dtype=torch.float32)

                # normalized_loss_values = sample_loss - epoch_loss_mean  # ?

        # Validation loop
        if ((step_idx + 1) % steps_per_epoch == 0) and (val_dataloader is not None):
            accelerator.print(f"## Validation - step: {step_idx} ##")
            model.eval()
            cumulated_loss = 0
            all_preds = []
            all_targets = []
            for batch in val_dataloader:
                with torch.no_grad():
                    logits, preds, probs = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["span_start"],
                        batch["span_end"],
                    )
                    loss = loss_fn(logits, batch["label"].long())
                    gathered_loss = accelerator.gather(loss)
                    cumulated_loss += gathered_loss.sum()
                preds, targets = accelerator.gather_for_metrics(
                    (preds, batch["label"].long())
                )

                if (num_classes > 2) and binarize_validation:
                    preds = torch.where(preds > 1, 0, preds)

                for metric in valid_metrics.values():
                    metric(preds, targets)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

            # Log loss
            accelerator.log({"valid/loss": cumulated_loss.item()}, step=step_idx)
            # Log to output
            accelerator.print(f"valid/loss: {cumulated_loss.item()}")

            # Validation Metrics
            valid_metric_results = {}
            for metric_name, metric in valid_metrics.items():
                metric_result = metric.compute().item()
                valid_metric_results[metric_name] = metric_result

                # Log to output
                accelerator.print(f"valid/{metric_name} : {metric_result}")

                # Log to tensorboard
                accelerator.log({f"valid/{metric_name}": metric_result}, step=step_idx)

                # Reseting internal state such that metric ready for new data
                metric.reset()

            if accelerator.is_main_process:
                all_preds = torch.cat(all_preds).cpu()
                all_targets = torch.cat(all_targets).cpu()

                f1 = f1_score(all_preds, all_targets, task="binary")
                p = precision(all_preds, all_targets, task="binary")

                accelerator.print(f"valid/precision_manual : {p}")
                accelerator.print(f"valid/f1_manual : {f1}")

            model.train()
            accelerator.print("## Training ##")

    accelerator.end_training()
    accelerator.print("Last step_idx", step_idx)

    # Save state dict
    if path_save_model is not None:
        save_model(
            accelerator=accelerator,
            model=model,
            experience_name=experience_name,
            path_save_model=path_save_model,
        )

    else:
        accelerator.print("### Training completed without saving")

    path_moving_loss = path_moving_loss.format(experience_name=experience_name)
    path_loss_values = path_loss_values.format(experience_name=experience_name)

    path_dir = os.path.dirname(path_loss_values)
    if accelerator.is_main_process:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

    accelerator.wait_for_everyone()

    accelerator.save(moving_loss, path_moving_loss)
    accelerator.print(f"### moving_loss saved at : {path_moving_loss}")

    accelerator.save(loss_values, path_loss_values)
    accelerator.print(f"### loss_values saved at : {path_loss_values}")


if __name__ == "__main__":
    app()
