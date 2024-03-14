import os
from pathlib import Path
from typing import Optional

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
    path_model: str,
    path_save_metrics: Optional[str] = None,
    batch_size: int = 64,  # effective batch size
    num_classes: int = 3,
    path_tokenizer: str = "/data/scratch/cse/camembert-EDS/",
    path_dataset: str = "/data/scratch/cse/oeci-ml/data/config_base/dataset_test/",
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
    dataset_test = TextDataset(path_dataset, tokenizer, split="test")

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
        model = torch.load(path_model, map_location=torch.device(device))

    # Prepare for ressources
    test_dataloader = accelerator.prepare(test_dataloader)

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
    if test_dataloader is not None:
        valid_metric_results = validate_model(
            model=model,
            val_dataloader=test_dataloader,
            accelerator=accelerator,
            valid_metrics=valid_metrics,
            step_idx=0,
            num_classes=num_classes,
            binarize_validation=binarize_validation,
            loss_fn=None,
            log=False,
            split="test",
        )

    # accelerator.end_training()

    if path_save_metrics is None:
        path_model = Path(path_model)
        path_save_metrics = path_model.joinpath(path_model.parent, "metrics.csv")
    df_valid_metric_results = pd.DataFrame(valid_metric_results, index=[0])
    df_valid_metric_results.to_csv(path_save_metrics, index=False)
    accelerator.print(f"Metrics saved at: {path_save_metrics}")


if __name__ == "__main__":
    app()
