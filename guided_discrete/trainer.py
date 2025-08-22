import os 
import time 
import torch 
import wandb
import numpy as np 

import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import Callback 
from pytorch_lightning.utilities import grad_norm 

from guided_discrete.sample import test_solving 
from collections import defaultdict
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

    def on_before_optimizer_step(self, optimizer):
        
        norms = self.get_gradient_norms()
        self.log_dict(norms)

    def get_gradient_norms(self):
        '''
        Returns a dictionary that contians all the layer norms and also 
            the average of all the layers for each network (ex: LayerNorm vs cls vs embeddings vs encoder)
        '''
        norms = {}
        network_norms = defaultdict(list)
        for name, param in self.named_parameters():
            if param.grad is None:
                continue 
            grad_norm_val = param.grad.norm().item()

            network_name = name.split(".")[1]
            network_norms[network_name].append(grad_norm_val)
            norms[name] = param.grad.norm().item()
        
        for network_name in network_norms:
            norms[f"{network_name}_avg"] = np.mean(network_norms[network_name])
        return norms 
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
        self.num_solutions_generate = config.num_solutions_generate
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch == 0:
            return 
            #return TODO: uncommment
             
        if pl_module.current_epoch % self.sample_frequency != 0:
            return 
        
        pl_module.eval()
        test_solving(pl_module, self.num_samples, self.num_solutions_generate, self.infill_dataset, self.vocab_file,\
                     self.exp_dir, pl_module.current_epoch, self.guidance_kwargs)
        print("We evaluated some solutions")
 



def get_trainer(config, num_train_batches):
    '''
    Returns a PL trainer for either evaluation or training mode 
    '''
            
    if config.is_eval:
        return pl.Trainer(
            default_root_dir=config['exp_dir'],
            devices=1,
            enable_progress_bar=True,
        )

    # finds the save directory that it should save to 
    sweep_iter = 0
    while True:
        save_valid_dir = os.path.join(config.exp_dir, f"models/best_by_valid_{sweep_iter}")
        save_train_dir = os.path.join(config.exp_dir, f"models/best_by_train_{sweep_iter}")
        if os.path.exists(save_train_dir):
            sweep_iter += 1  
        else:
            os.makedirs(save_train_dir)
            os.makedirs(save_valid_dir)
            break 

    callbacks= [pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=save_valid_dir,
            save_top_k=5,
            mode="min"
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='train_loss',
            dirpath=save_train_dir,
            save_top_k=5,
            mode="min"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        LossLoggingCallback(config.exp_dir),
        ]
    if config.is_sudoku:
        callbacks.append(SampleEvaluationCallback(config))
        

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
        enable_progress_bar=True,
        #limit_train_batches=1, #OVERFIT TODO: delete
        #limit_val_batches=1
    )
    return trainer 

    



