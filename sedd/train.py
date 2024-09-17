"""Training and evaluation"""
import yaml 
import os
import numpy as np
import run_train
from datetime import datetime
import sys
import os 
import argparse 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main(cfg):
    run_train._run(cfg)
    
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'configs/normal_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        now = datetime.now()

        # add working directory 
        #work_dir = 'experiments/09-03-2024-14:45'
        work_dir = os.path.join('experiments', now.strftime("%m-%d-%Y-%H:%M"))
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        config['work_dir'] = work_dir
        main(config)