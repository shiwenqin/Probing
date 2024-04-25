import wandb
from omegaconf import OmegaConf

class WandbLogger:
    def __init__(self, cfg):
        self.active = cfg.wandb.active
        if not self.active:
            return

        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=config_dict,
        )
    
    def log(self, metrics):
        if not self.active:
            return
        wandb.log(metrics)

class StatTracker:
    # Track and Calculate statistics
    def __init__(self):
        self.loss = 0
        self.acc = 0
        self.num = 0
        self.f1 = 0
        self.batch_num = 0

    def update(self, loss, acc, num, f1):
        self.loss += loss
        self.acc += acc
        self.num += num
        self.f1 += f1   
        self.batch_num += 1

    def get_stats(self):
        loss = self.loss / self.num
        acc = self.acc / self.num
        f1 = self.f1 / self.batch_num
        return loss, acc, f1

class EarlyStopper:
    # Standard early stopper
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
