from typing import Any, Optional

import torch
from torchmetrics.functional import f1_score, precision


def validate_model(
    model,
    val_dataloader,
    accelerator,
    valid_metrics,
    step_idx: int,
    num_classes: int,
    binarize_validation: bool,
    loss_fn: Optional[Any] = None,
    log: bool = True,
    split: str = "valid",
):
    accelerator.print(f"## {split} - step: {step_idx} ##")
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
            if loss_fn is not None:
                loss = loss_fn(logits, batch["label"].long())
                gathered_loss = accelerator.gather(loss)
                cumulated_loss += gathered_loss.sum()
        preds, targets = accelerator.gather_for_metrics((preds, batch["label"].long()))

        if (num_classes > 2) and binarize_validation:
            preds = torch.where(preds > 1, 0, preds)

        for metric in valid_metrics.values():
            metric(preds.cpu(), targets.cpu())

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    if loss_fn is not None:
        if log:
            # Log loss
            accelerator.log({f"{split}/loss": cumulated_loss.item()}, step=step_idx)
        # Log to output
        accelerator.print(f"{split}/loss: {cumulated_loss.item()}")

    # Reset the model to train
    model.train()

    # Validation Metrics
    if accelerator.is_main_process:
        valid_metric_results = {}
        for metric_name, metric in valid_metrics.items():
            metric_result = metric.compute().item()
            valid_metric_results[metric_name] = metric_result

            # Print to output
            accelerator.print(f"{split}/{metric_name} : {metric_result}")

            # Log to tensorboard
            if log:
                accelerator.log(
                    {f"{split}/{metric_name}": metric_result}, step=step_idx
                )

            # Reseting internal state such that metric ready for new data
            metric.reset()
        all_preds = torch.cat(all_preds).cpu()
        all_targets = torch.cat(all_targets).cpu()

        f1 = f1_score(all_preds, all_targets, task="binary")
        p = precision(all_preds, all_targets, task="binary")

        accelerator.print(f"{split}/precision_manual : {p}")
        accelerator.print(f"{split}/f1_manual : {f1}")
        accelerator.print("## Training ##")
        return valid_metric_results
