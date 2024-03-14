def log_training_step(
    accelerator,
    loss_all_devices,
    step_idx,
    total_steps,
    optimizer,
    accu_loss,
    log_each_device,
):
    # Log to output
    accelerator.print(
        f"loss: {loss_all_devices.item():>7f}   [{step_idx:>5d}/{total_steps:>5d}]"
    )

    # Get lr
    lr_transformer = optimizer.optimizer.param_groups[0]["lr"]
    lr_classification_head = optimizer.optimizer.param_groups[1]["lr"]

    # Log to tensorboard
    accelerator.log(
        {
            "train/loss": loss_all_devices.item(),
            "lr_transformer": lr_transformer,
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
