import os 
import time 
import torch 
import wandb

import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import Callback 

from seq_models.sample import sample_model 

class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()
    
    def training_step(self, batch):
        out = self.forward(
            batch['seq'],
            batch['corrupt_mask'],
            batch['attn_mask'],
            None,
            return_by_timestep=True 
        )

        log_dict = {f"train_{k}":v for k,v in out.items()}
        self.log_dict(log_dict)

        return out['loss']
    
    def configure_optimizers(self):
        config = {
            'optimizer': self.opt 
        }
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

            config['lr_scheduler'] = {
                "scheduler": self.lr_scheduler,
                "frequency":1,
                "interval": "epoch"
            }

        return config 

#TODO: finish this
class SampleEvaluationCallback(Callback):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        log_dir = os.path.join(self.config['exp_dir'], "samples" )
        os.path.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'sample.txt')
        self.log_file = log_file 
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            return 
        
        if pl_module.current_epoch % self.sample_frequency != 0:
            return 
        
        pl_module.eval()


def get_trainer(config, num_train_batches):
    os.makedirs(os.path.join(config.exp_dir, "models/best_by_vaid"), exist_ok=True)
    os.makedirs(os.path.join(config.exp_dir, "models/best_by_train"), exist_ok=True)

    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join(config.exp_dir, "models/best_by_valid"),
            save_top_k=5,
            mode="min"
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='train_loss',
            dirpath=os.path.join(config.exp_dir, "models/best_by_train"),
            save_top_k=5,
            mode="min"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        #TODO: add sample evaluation callback
    ]

    wandb_logger = WandbLogger(project="guided_seq", dir=config['exp_dir'])
    accelerator, strategy = "cpu", None 
    if torch.cuda_is_available():
        accelerator = "gpu"
        strategy = "ddp"
    
    trainer = pl.Trainer(
        default_root_dir=config['exp_dir'],
        gradient_clip_val=config['gradient_clip'],
        min_epochs=config['min_epochs'],
        max_epochs=config['max_epochs'],
        check_val_every_n_epochs=1, 
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=min(200, num_train_batches),
        accelerator=accelerator,
        strategy=strategy,
        devices=config['ngpu'],
        enable_progress_bar=True
    )
    return trainer 

    



