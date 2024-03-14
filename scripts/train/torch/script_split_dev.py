import os
from datetime import datetime
from itertools import count
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from confit import Cli
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Precision, Recall

from oeciml.torchmodules.data.dataset import (
    TextDataset,
    TrainValidtoDatasetDict,
    collate_fn,
)
from oeciml.torchmodules.helpers.misc import create_directory, save_model
from oeciml.torchmodules.logger import log_training_step
from oeciml.torchmodules.loss import PytorchLoss
from oeciml.torchmodules.model import (
    QualifierModelv2,
    ToyModel,
    tokenizer_from_pretrained,
)
from oeciml.torchmodules.optimizer import LinearSchedule, ScheduledOptimizer
from oeciml.torchmodules.trainer import validate_model

app = Cli(pretty_exceptions_show_locals=False)


@app.command(name="script")
def main(
    num_accumulation_steps: int = 1,  # Gradient accumulation
    batch_size: int = 128,  # effective batch size
    total_steps: Optional[int] = None,
    n_epochs: Optional[int] = None,
    log_each_n_steps: int = 10,
    log_each_device: bool = False,
    path_save_model: Optional[str] = None,  # folder
    optimizer_params: Dict[str, Any] = dict(
        head=dict(lr=5e-4, warmup_rate=0),
        transformer=dict(lr=5e-5, warmup_rate=0.1),
    ),
    num_classes: int = 3,
    loss_fn: _Loss = PytorchLoss("CrossEntropyLoss"),
    experience_name: str = "ToyModel",
    path_embedding: str = "/data/scratch/cse/camembert-EDS",
    path_tensorboard: str = "/export/home/cse/tensorboard_data/",
    path_tokenizer: str = "/data/scratch/cse/camembert-EDS/",
    path_dataset_tmp: str = "/data/scratch/cse/oeci-ml/data/config_base/tmp/",
    path_valid_set: str = "",  # pd.DataFrame
    path_test_set: str = "",
    path_save_metrics: Optional[str] = None,
    n_patients: int = 40,
    binarize_validation: bool = True,
    mask_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = 54,
    debug: bool = False,
):
    if seed is None:
        seed = np.random.randint(0, 1000)
    set_seed(seed)
    assert bool(total_steps) != bool(n_epochs)

    # Set env variables
    torch.cuda.empty_cache()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Tracker (tensorboard)
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

    # Tokenizer
    tokenizer = tokenizer_from_pretrained(path=path_tokenizer)

    # Path dataset
    path_dataset_tmp = os.path.join(path_dataset_tmp, experience_name)
    # Data
    if accelerator.is_main_process:
        set_seed(seed)
        valid_labeled = pd.read_pickle(path_valid_set)
        assert n_patients <= valid_labeled.person_id.nunique()
        test_data = pd.read_pickle(path_test_set)

        train_ids = valid_labeled.person_id.drop_duplicates().sample(n_patients)
        # n_entities = len(valid_labeled.loc[valid_labeled.person_id.isin(train_ids)])

        dataset_creator = TrainValidtoDatasetDict(
            text_col="masked_span",
            id_col="text_id",
            span_cols=["masked_span_start_char", "masked_span_end_char"],
            label_col="label",
            balance_to_class=-1,
            save_path=path_dataset_tmp,
        )
        _ = dataset_creator.generate(
            train_data=valid_labeled.loc[valid_labeled.person_id.isin(train_ids)],
            validation_data=valid_labeled.loc[~valid_labeled.person_id.isin(train_ids)],
            test_data=test_data,
        )
    accelerator.wait_for_everyone()

    dataset_train = TextDataset(
        path_dataset_tmp, tokenizer, split="train", mask_params=mask_params
    )
    n_entities = len(dataset_train)
    set_seed(seed)
    dataset_valid = TextDataset(path_dataset_tmp, tokenizer, split="val")
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

    if len(dataset_valid) > 0:
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
    dataset_test = TextDataset(path_dataset_tmp, tokenizer, split="test")

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

    # Model
    set_seed(seed)
    if debug:
        model = ToyModel(num_classes=num_classes, num_embedding=len(tokenizer))
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
Train dataset length {len(train_dataloader) *  batch_size}
N patients {n_patients}
N entities {n_entities}

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
        "loss": loss_fn,
        "num_classes": num_classes,
        "path_dataset": path_dataset_tmp,
        "experience_name": experience_name,
        "path_tokenizer": path_tokenizer,
        "mask_params": mask_params,
        "seed": seed,
        "n_patients": n_patients,
        "n_entities": n_entities,
    }
    hps["optimizer_params"] = optimizer_params

    accelerator.init_trackers("")
    if accelerator.is_main_process:
        with open(os.path.join(path_tensorboard, "hparams.yml"), "w") as outfile:
            try:
                yaml.dump(hps, outfile)
            except yaml.representer.RepresenterError:
                raise

    # Train loop
    set_seed(seed)
    model.train()
    train_iterator = (batch for epoch in count() for batch in train_dataloader)
    accelerator.print("## Training ##")
    for step_idx in range(total_steps):
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
                accu_loss += loss.detach()

                # Backpropagation
                accelerator.backward(loss)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

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

    if test_dataloader is not None:
        test_metric_results = validate_model(
            model=model,
            val_dataloader=test_dataloader,
            accelerator=accelerator,
            valid_metrics=valid_metrics,
            step_idx=step_idx,
            num_classes=num_classes,
            binarize_validation=binarize_validation,
            loss_fn=None,
            log=True,
            split="test",
        )

    accelerator.wait_for_everyone()
    if (path_save_metrics is not None) and (accelerator.is_main_process):
        path_save_metrics = create_directory(
            accelerator=accelerator,
            experience_name=experience_name,
            path_save_model=path_save_metrics,
            file_name="metrics.csv",
        )

        df_valid_metric_results = pd.DataFrame(test_metric_results, index=[0])
        df_valid_metric_results["n_patients"] = n_patients
        df_valid_metric_results["n_entities"] = n_entities
        df_valid_metric_results.to_csv(path_save_metrics, index=False)
        accelerator.print(f"Metrics saved at: {path_save_metrics}")

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


if __name__ == "__main__":
    app()
