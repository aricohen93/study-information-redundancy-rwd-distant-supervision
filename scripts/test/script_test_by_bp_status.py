import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from confit import Cli
from torch.utils.data import DataLoader
from torchmetrics import F1Score, Precision, Recall

from oeciml.torchmodules.data.dataset import TextDataset, collate_fn
from oeciml.torchmodules.model import ToyModel, tokenizer_from_pretrained
from oeciml.torchmodules.trainer import validate_model

app = Cli(pretty_exceptions_show_locals=False)

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.command(name="script")
def main(
    path_models: List[str],
    path_save_metrics: Optional[str] = None,
    batch_size: int = 64,  # effective batch size
    num_classes: int = 3,
    path_tokenizer: str = "/data/scratch/cse/camembert-EDS/",
    path_dataset: str = "/data/scratch/cse/oeci-ml/data/config_base/dataset_test_by_bp_status/",
    binarize_validation: bool = True,
    seed: int = 54,
    debug: bool = False,
):
    set_seed(seed)

    # Set env variables
    torch.cuda.empty_cache()

    # Accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        kwargs_handlers=[kwargs],
    )

    device = accelerator.device
    accelerator.print(f"Device : {device} num_processes: {accelerator.num_processes}")

    # Tokenizer
    tokenizer = tokenizer_from_pretrained(path=path_tokenizer)

    # Data
    set_seed(seed)
    ds_w_bp = TextDataset(path_dataset, tokenizer, split="test_with_bp")
    ds_wo_bp = TextDataset(path_dataset, tokenizer, split="test_without_bp")

    set_seed(seed)
    w_bp_dataloader = DataLoader(
        ds_w_bp,
        batch_size=batch_size // (accelerator.num_processes),
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    wo_bp_dataloader = DataLoader(
        ds_wo_bp,
        batch_size=batch_size // (accelerator.num_processes),
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    for path_model in path_models:
        accelerator.print(path_model)
        set_seed(seed)
        if debug:
            model = ToyModel(num_classes=num_classes, num_embedding=len(tokenizer))
        else:
            model = torch.load(path_model, map_location=torch.device(device))

        # Prepare for ressources
        w_bp_dataloader, wo_bp_dataloader = accelerator.prepare(
            w_bp_dataloader, wo_bp_dataloader
        )

        model = accelerator.prepare(model)

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

        # Test loop
        if w_bp_dataloader is not None:
            w_bp_metric_results = validate_model(
                model=model,
                val_dataloader=w_bp_dataloader,
                accelerator=accelerator,
                valid_metrics=valid_metrics,
                step_idx=0,
                num_classes=num_classes,
                binarize_validation=binarize_validation,
                loss_fn=None,
                log=False,
                split="with BP",
            )

        if wo_bp_dataloader is not None:
            wo_bp_metric_results = validate_model(
                model=model,
                val_dataloader=wo_bp_dataloader,
                accelerator=accelerator,
                valid_metrics=valid_metrics,
                step_idx=0,
                num_classes=num_classes,
                binarize_validation=binarize_validation,
                loss_fn=None,
                log=False,
                split="without BP",
            )

        # accelerator.end_training()

        if path_save_metrics is None:
            path_model = Path(path_model)
            path_save_metrics = path_model.joinpath(
                path_model.parent, "metrics_by_bp_status.csv"
            )
        df_valid_metric_results_w_bp = pd.DataFrame(w_bp_metric_results, index=[0])
        df_valid_metric_results_w_bp["status"] = "w_bp"
        df_valid_metric_results_wo_bp = pd.DataFrame(wo_bp_metric_results, index=[0])
        df_valid_metric_results_wo_bp["status"] = "wo_bp"

        df_valid_metric_results = pd.concat(
            [df_valid_metric_results_w_bp, df_valid_metric_results_wo_bp]
        )

        df_valid_metric_results.to_csv(path_save_metrics, index=False)
        accelerator.print(f"Metrics saved at: {path_save_metrics}")
        path_save_metrics = None


if __name__ == "__main__":
    app()
