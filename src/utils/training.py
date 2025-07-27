from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR


def scheduler_step(scheduler, metric, epoch_idx):
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric)
    elif isinstance(scheduler, LambdaLR):
        scheduler.step(epoch_idx)
    else:
        scheduler.step()