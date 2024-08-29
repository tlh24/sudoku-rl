"""Training and evaluation"""
import yaml 
import os
import numpy as np
import run_train
import utils
from datetime import datetime

def main(cfg):
    ngpus = cfg.ngpus
	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    run_train._run(cfg)
    

if __name__ == "__main__":
    with open('configs/normal_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        now = datetime.now()

        # add working directory 
        work_dir = os.path.join('experiments', now.strftime("%m-%d-%Y-%H:%M"))
        config['work_dir'] = work_dir
        main(config)