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

from oeciml.torchmodules.data.dataset import TextDataset, collate_fn
from oeciml.torchmodules.helpers.misc import (
    compute_metrics,
    create_directory,
    save_model,
)
from oeciml.torchmodules.logger import log_training_step
from oeciml.torchmodules.loss import PytorchLoss
from oeciml.torchmodules.model import (
    QualifierModelv2,
    ToyModel,
    tokenizer_from_pretrained,
)
from oeciml.torchmodules.optimizer import LinearSchedule, ScheduledOptimizer
from oeciml.torchmodules.sample_selection.lrt.flip_labels_lrt import lrt_flip_scheme
from oeciml.torchmodules.trainer import validate_model

# from datasets import DatasetDict

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="script")
def main(
    window_size: int = 2,  # rolling windows to get eta_tilde
    epoch_start_update: int = 1,  # the epoch to start correction (warm-up period)
    epoch_start_second_loss: int = 1,
    delta_base: float = 1.2,
    incremental_delta_by_epoch: float = 0.02,
    num_accumulation_steps: int = 1,  # Gradient accumulation
    batch_size: int = 128,  # effective batch size
    total_steps: Optional[int] = None,
    n_epochs: Optional[int] = None,
    log_each_n_steps: int = 10,
    log_each_device: bool = False,
    path_save_model: Optional[str] = None,
    optimizer_params: Dict[str, Any] = dict(
        head=dict(lr=5e-4, warmup_rate=0),
        transformer=dict(lr=5e-5, warmup_rate=0.1),
    ),
    num_classes: int = 3,
    loss_fn: _Loss = PytorchLoss("CrossEntropyLoss"),
    loss_fn2: _Loss = PytorchLoss("CrossEntropyLoss"),
    experience_name: str = "LRTToyModel",
    path_embedding: str = "/data/scratch/cse/camembert-EDS",
    path_tensorboard: str = "/export/home/cse/tensorboard_data/",
    path_tokenizer: str = "/data/scratch/cse/camembert-EDS/",
    path_train_dataset: str = "/data/scratch/cse/oeci-ml/data/config_base/dataset_config_PR_SP_PS/",
    path_dev_dataset: Optional[
        str
    ] = "/data/scratch/cse/oeci-ml/data/config_base/dataset_dev/",
    path_save_dataset: Optional[str] = None,
    path_test_dataset: Optional[str] = None,
    path_save_metrics: Optional[str] = "/data/scratch/cse/models/",
    binarize_validation: bool = True,
    mask_params: Optional[Dict[str, Any]] = None,
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
    delta_base_str = str(delta_base).replace(".", "-")
    experience_name = experience_name + "_delta_" + delta_base_str

    if mask_params is not None:
        forget_rate_str = str(mask_params["forget_rate"])
        experience_name = experience_name + "_" + forget_rate_str[2:]

    experience_name = os.path.join(
        experience_name,
        datetime.today().strftime("%Y_%m_%d_T_%H_%M_%S"),
    )

    path_tensorboard = os.path.join(path_tensorboard, experience_name)

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
    dataset_train = TextDataset(
        path_train_dataset, tokenizer, split="train", mask_params=mask_params
    )
    set_seed(seed)
    dataset_valid = TextDataset(path_train_dataset, tokenizer, split="val")

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

    set_seed(seed)
    if path_test_dataset is not None:
        dataset_test = TextDataset(path_test_dataset, tokenizer, split="test")

        set_seed(seed)
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=batch_size // (accelerator.num_processes),
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        test_dataloader = None

    # Model
    set_seed(seed)
    if debug:
        model = ToyModel(num_classes=num_classes)
    else:
        model = QualifierModelv2(
            path_embedding=path_embedding,
            num_classes=num_classes,
            **dict(tokenizer=tokenizer),
        )

    # Prepare for ressources
    (train_dataloader, val_dataloader, test_dataloader) = accelerator.prepare(
        train_dataloader, val_dataloader, test_dataloader
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

    optimizer = ScheduledOptimizer(
        torch.optim.AdamW(
            [
                {
                    "params": model.transformer.parameters(),
                    "lr": optimizer_params["transformer"]["lr"],
                    "schedules": LinearSchedule(
                        path="lr",
                        warmup_rate=optimizer_params["transformer"]["warmup_rate"],
                        total_steps=optimizer_params["total_steps"],
                    ),
                },
                {
                    "params": model.classification_head.parameters(),
                    "lr": optimizer_params["head"]["lr"],
                    "schedules": LinearSchedule(
                        path="lr",
                        warmup_rate=optimizer_params["head"]["warmup_rate"],
                        total_steps=optimizer_params["total_steps"],
                    ),
                },
            ]
        )
    )
    optimizer, model = accelerator.prepare(optimizer, model)

    # # Metrics
    # valid_metrics = {
    #     "f1": F1Score(task="binary"),
    #     "precision": Precision(task="binary"),
    #     "recall": Recall(task="binary"),
    # }
    # for metric_name, metric in valid_metrics.items():
    #     valid_metrics[metric_name] = metric.to(accelerator.device)  # todo prepare ?

    # Metrics
    valid_metrics = {
        "f1": F1Score(task="binary", compute_on_cpu=True, sync_on_compute=False),
        "precision": Precision(
            task="binary", compute_on_cpu=True, sync_on_compute=False
        ),
        "recall": Recall(task="binary", compute_on_cpu=True, sync_on_compute=False),
    }
    for metric_name, metric in valid_metrics.items():
        # valid_metrics[metric_name] = metric.to(accelerator.device)  # todo prepare ?
        metric.sync_context(should_sync=False)
        valid_metrics[metric_name] = metric.cpu()  # todo prepare ?

    # Tracker (Tensorboard and hparams)
    hps = {
        "num_iterations": total_steps,
        "batch_size": batch_size,
        "delta_base": delta_base,
        "num_classes": num_classes,
        "window_size": window_size,
        "epoch_start_update": epoch_start_update,
        "epoch_start_second_loss": epoch_start_second_loss,
        "path_dataset": path_train_dataset,
    }
    hps["optimizer_params"] = optimizer_params
    accelerator.init_trackers("")
    if accelerator.is_main_process:
        with open(os.path.join(path_tensorboard, "hparams.yml"), "w") as outfile:
            try:
                yaml.dump(hps, outfile)
            except yaml.representer.RepresenterError:
                raise

    # LRT
    ntrain = len(dataset_train)
    pred_softlabels = torch.zeros(
        [ntrain, window_size, num_classes], dtype=torch.float32
    )
    count_labels = torch.zeros(num_classes, dtype=torch.long)

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
            delta = delta_base + incremental_delta_by_epoch * max(
                epoch - epoch_start_update, 0
            )
            count_labels = torch.zeros(num_classes, dtype=torch.long)

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
                if epoch >= epoch_start_second_loss:
                    loss = loss_fn2(logits, batch["label"].long())
                else:
                    loss = loss_fn(logits, batch["label"].long())

                # Loss (gradient accumulation)
                accu_loss += loss.detach()

                # Backpropagation
                accelerator.backward(loss)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Count labels
                bincount = torch.bincount(
                    accelerator.gather_for_metrics(batch["label"].long()).cpu(),
                    minlength=num_classes,
                )
                count_labels = torch.vstack([count_labels, bincount]).sum(axis=0)

            # window_size : rolling windows to get eta_tilde
            if epoch >= (epoch_start_update - window_size):
                indices, probs = accelerator.gather_for_metrics(
                    (batch["id"].flatten(), probs)
                )
                indices = indices.cpu().numpy()
                probs = probs.detach().cpu()
                pred_softlabels[indices, epoch % window_size, :] = probs

        # Loss of all devices
        loss_all_devices = accelerator.gather(accu_loss).sum()

        # Log (training steps)
        if step_idx % log_each_n_steps == 0:
            log_training_step(
                accelerator=accelerator,
                loss_all_devices=loss_all_devices,
                step_idx=step_idx,
                total_steps=total_steps,
                optimizer=optimizer,
                accu_loss=accu_loss,
                log_each_device=log_each_device,
            )

        # applying lRT scheme
        # args_epoch_update : epoch to update labels
        if (step_idx + 1) % steps_per_epoch == 0:
            accelerator.print("count labels", count_labels)
            if epoch >= epoch_start_update:
                accelerator.wait_for_everyone()
                accelerator.print("### UPDATE labels ###")
                accelerator.print(
                    "Pred soft_labels",
                    pred_softlabels.device,
                    "non zero",
                    pred_softlabels.count_nonzero(),
                    "sum",
                    pred_softlabels.sum(),
                )

                if accelerator.is_main_process:
                    y_tilde = dataset_train.get_data_labels().flatten().long()
                    pred_softlabels_bar = pred_softlabels.mean(1)
                    accelerator.print("> delta:", delta)
                    new_labels, clean_softlabels, changed_idx = lrt_flip_scheme(
                        pred_softlabels_bar, torch.clone(y_tilde), delta
                    )
                    new_labels = clean_softlabels.argmax(1)
                    for idx in changed_idx:
                        accelerator.log(
                            {
                                f"text old_label: {y_tilde[idx]}, new_label: {new_labels[idx]}": dataset_train[
                                    idx
                                ][
                                    "text"
                                ]
                            },
                            step=step_idx,
                        )
                    dataset_train.update_corrupted_label(new_labels)

                    n_changed_labels = (new_labels != y_tilde).sum().item()
                    accelerator.print(f"number of changed labels: {n_changed_labels}")

        # Validation loop
        if (
            (((step_idx + 1) % steps_per_epoch) == 0) or ((step_idx + 1) == total_steps)
        ) and (val_dataloader is not None):
            validate_model(
                model=model,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
                valid_metrics=valid_metrics,
                step_idx=step_idx,
                num_classes=num_classes,
                binarize_validation=binarize_validation,
                loss_fn=loss_fn,
            )
            model.train()

    # Test metrics
    compute_metrics(
        test_dataloader=test_dataloader,
        model=model,
        accelerator=accelerator,
        valid_metrics=valid_metrics,
        step_idx=step_idx,
        num_classes=num_classes,
        binarize_validation=binarize_validation,
        experience_name=experience_name,
        path_save_metrics=path_save_metrics,
        extra_info_to_save={
            "experience_name": experience_name,
            "path_dataset": path_train_dataset,
            "path_test_dataset": path_test_dataset,
            "total_steps": total_steps,
            "seed": seed,
            "delta_base": delta_base,
            "epoch_start_update": epoch_start_update,
            "window_size": window_size,
            "epoch_start_second_loss": epoch_start_second_loss,
        },
    )

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

    if path_save_dataset is not None:
        accelerator.wait_for_everyone()
        path_save_dataset = create_directory(
            accelerator, experience_name, path_save_dataset, file_name="dataset"
        )

        # accelerator.save(dataset_train, path_save_dataset)
        if accelerator.is_main_process:
            pass  # FIXME
            # dataset_train.data.save_to_disk(path_save_dataset)  # FIXME
            # DatasetDict({"train": dataset_train.data}).save_to_disk(path_save_dataset)
        # accelerator.print(f"### Dataset saved at : {path_save_dataset}")


if __name__ == "__main__":
    app()
