import os 
import time 
import torch 
import wandb

import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import Callback 

from guided_discrete.sample import test_solving 

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
        self.log('training_loss', out['loss'], on_step=False, on_epoch=True, prog_bar=True)

        return out['loss']

    def validation_step(self, batch):
        with torch.no_grad():
            out = self.forward(
                batch["seq"],
                batch["corrupt_mask"],
                batch["attn_mask"],
                labels=batch["labels"] if "labels" in batch else None,
                return_by_timestep=True
            )
        log_dict = {f"val_{k}":v for k,v in out.items()}
        self.log_dict(log_dict)
        self.log('validation_loss', out['loss'], on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": out["loss"]}
    
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

class LossLoggingCallback(Callback):
    '''
    Write down train and val loss to text file 
    '''
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'loss_log.txt')
    
    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.log_dir, exist_ok=True)

        with open(self.log_file, 'a+') as f:
            f.write("Epoch, Train Loss, Val Loss, Train Accuracy, Val Accuracy\n")
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get('training_loss', float('nan'))
        val_loss = trainer.callback_metrics.get('validation_loss', float('nan'))
        train_accuracy = trainer.callback_metrics.get('train_accuracy', float('nan'))
        val_accuracy = trainer.callback_metrics.get('val_accuracy', float('nan'))

        with open(self.log_file, 'a') as f:
            f.write(f"{epoch}, {train_loss:4f},{val_loss:.4f},{train_accuracy:4f},{val_accuracy:4f}\n")

    def on_fit_end(self, trainer, pl_module):
        print(f"Loss log saved to {self.log_file}")


class SampleEvaluationCallback(Callback):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        self.exp_dir = self.config['exp_dir']
        self.vocab_file = self.config.vocab_file 
        self.num_samples = self.config.num_samples
        self.sample_frequency = config.val_sample_frequency
        self.guidance_kwargs = config.guidance_kwargs 
        self.infill_dataset = config.infill_dataset 
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            return 
            #return TODO: uncommment
        
        if pl_module.current_epoch % self.sample_frequency != 0:
            return 
        
        pl_module.eval()
      
        test_solving(pl_module, self.num_samples, self.infill_dataset,self.vocab_file,\
                     self.exp_dir, self.guidance_kwargs)
        print("We evaluated some solutions")
 



def get_trainer(config, num_train_batches):
    os.makedirs(os.path.join(config.exp_dir, "models/best_by_valid"), exist_ok=True)
    os.makedirs(os.path.join(config.exp_dir, "models/best_by_train"), exist_ok=True)
    
    callbacks= [pl.callbacks.ModelCheckpoint(
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
        LossLoggingCallback(config.exp_dir),
        SampleEvaluationCallback(config)
        ]

    if config.use_wandb:
        logger = WandbLogger(project="guided_seq", dir=config['exp_dir'])
    else:
        logger = None 

    accelerator, strategy = "cpu", None 
    if torch.cuda.is_available():
        accelerator = "gpu"
        strategy = "ddp"
    
    trainer = pl.Trainer(
        default_root_dir=config['exp_dir'],
        gradient_clip_val=config['gradient_clip'],
        min_epochs=config['min_epochs'],
        max_epochs=config['max_epochs'],
        check_val_every_n_epoch=1, 
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=min(200, num_train_batches),
        accelerator=accelerator,
        strategy=strategy,
        devices=config['ngpu'],
        enable_progress_bar=True
    )
    return trainer 

    



