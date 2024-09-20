import sys 
import os 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import pprint 
from pathlib import Path 
import torch
import warnings 
from trainer import get_trainer
from omegaconf import OmegaConf
import hydra 
from utils import (flatten_config, convert_to_dict)
import wandb 
from data_utils import get_dataloader



@hydra.main(config_path="./configs", config_name="train_seq_model")
def main(config):
    Path(config.exp_dir).mkdir(parents=True, exist_ok=True) 
    # for logging purposes 
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep='/')
    log_config = {'/'.join(('config', key)): val for key, val in log_config.items()}
    if config.use_wandb:
        wandb.init(
            project="guided_seq",
            config=log_config
        )
    pprint.PrettyPrinter(depth=4).pprint(convert_to_dict(config))
    
    #config.model.network.target_channels = len(config.target_cols)
    # this initializes a model (ex: MLMDiffusion model) based on the parameters in config
    model = hydra.utils.instantiate(config.model, _recursive_=False)

    train_dl = get_dataloader(config, 'train')
    valid_dl = get_dataloader(config, 'validation')
    
    trainer = get_trainer(config, len(train_dl))

    trainer.fit(
        model=model, 
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl, 
        ckpt_path=config.resume_ckpt
    )

if __name__ == "__main__":
    main()
 

