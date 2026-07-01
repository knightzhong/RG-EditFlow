def should_save_epoch_checkpoint(epoch: int, save_every: int) -> bool:
    if save_every <= 0:
        return False
    return epoch % save_every == 0
