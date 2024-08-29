"""Training and evaluation"""
import yaml 
import os
import numpy as np
import run_train
from datetime import datetime

def main(cfg):
    run_train._run(cfg)
    
if __name__ == "__main__":
    with open('configs/normal_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        now = datetime.now()

        # add working directory 
        work_dir = os.path.join('experiments', now.strftime("%m-%d-%Y-%H:%M"))
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        config['work_dir'] = work_dir
        main(config)