import os
from typing import Any, Dict, Optional

import pandas as pd
import torch
from confit.config import Reference

from oeciml.torchmodules.trainer import validate_model


def set_proxy():
    proxy = "http://proxym-inter.aphp.fr:8080"
    os.environ["http_proxy"] = proxy
    os.environ["HTTP_PROXY"] = proxy
    os.environ["https_proxy"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            v = str(v) if isinstance(v, Reference) else v
            items.append((new_key, v))
    return dict(items)


def shift(x, dim, n, pad=0):
    shape = list(x.shape)
    shape[dim] = abs(n)

    slices = [slice(None)] * x.ndim
    slices[dim] = slice(n, None) if n >= 0 else slice(None, n)
    pad = torch.full(shape, fill_value=pad, dtype=x.dtype, device=x.device)
    x = torch.cat(
        ([pad] if n > 0 else []) + [x] + ([pad] if n < 0 else []), dim=dim
    ).roll(dims=dim, shifts=n)
    return x[tuple(slices)]


def create_directory(
    accelerator, experience_name, path_save_model, file_name="model.pt"
):
    path_save_model = os.path.join(path_save_model, experience_name, file_name)
    path_dir = os.path.dirname(path_save_model)
    if accelerator.is_main_process:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        return path_save_model


def save_model(
    accelerator, model, experience_name, path_save_model, model_name="model.pt"
):
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    # state_dict = model.state_dict()
    path_save_model = create_directory(
        accelerator=accelerator,
        experience_name=experience_name,
        path_save_model=path_save_model,
        file_name=model_name,
    )

    accelerator.print(f"### Model saved at : {path_save_model}")
    accelerator.save(model, path_save_model)


def compute_metrics(
    test_dataloader: Optional[Any],
    model: Any,
    accelerator: Any,
    valid_metrics: Any,
    step_idx: int,
    num_classes: int,
    binarize_validation: bool,
    experience_name: str,
    path_save_metrics: str,
    extra_info_to_save: Optional[Dict[str, Any]] = None,
):
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
    else:
        test_metric_results = None
        return test_metric_results

    accelerator.wait_for_everyone()
    if (test_metric_results is not None) and (accelerator.is_main_process):
        path_save_metrics = create_directory(
            accelerator=accelerator,
            experience_name=experience_name,
            path_save_model=path_save_metrics,
            file_name="metrics.csv",
        )
        if extra_info_to_save is not None:
            test_metric_results.update(extra_info_to_save)
        df_test_metric_results = pd.DataFrame(test_metric_results, index=[0])
        df_test_metric_results.to_csv(path_save_metrics, index=False)
        accelerator.print(f"Metrics saved at: {path_save_metrics}")

        return test_metric_results
