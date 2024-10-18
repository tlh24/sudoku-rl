import sys 
import os 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from omegaconf import OmegaConf

import hydra 
from utils import (flatten_config, convert_to_dict, count_parameters)
from data_utils import get_dataloader 
from sample import test_solving 
from pytorch_lightning import LightningModule 
import re 
import torch 

# NOTE: You must change the config path to be the saved model config!
@hydra.main(config_path="/home/justin/Desktop/Code/sudoku-rl/guided_discrete/logs/mlm_test/2024-10-17_12-20-19/.hydra", config_name="config")
def eval(config):
    # this initializes a model (ex: MLMDiffusion model) based on the parameters in config
    model = hydra.utils.instantiate(config.model, _recursive_=False)
    
    # this uses the evaluation config to ensure that we load the right eval parameters 
    saved_config = OmegaConf.load("/home/justin/Desktop/Code/sudoku-rl/guided_discrete/configs/eval_seq_model.yaml") 
    merged_config = OmegaConf.merge(saved_config, config)
    # TODO: find a fix to force priority to saved_config 
    merged_config.eval_ckpt = saved_config.eval_ckpt
    merged_config.num_eval_samples = saved_config.num_eval_samples
    merged_config.infill_dataset = saved_config.infill_dataset
    merged_config.is_eval = saved_config.is_eval

    if isinstance(model, LightningModule):
        checkpoint = torch.load(merged_config.eval_ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise NotImplementedError("Loading from pytorch models not implemented")
    
    # get the model directory from checkpoint 
    pattern = r'^(.*?/\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    match = re.search(pattern, merged_config.eval_ckpt)
    if match:
        model_exp_dir = match.group(1)     
    else:
        raise ValueError("Received a ckpt path that does not exist in the usual saved folder location")

    print(f"Number of params in the model: {count_parameters(model)}")
    
    solve_rate = test_solving(model, merged_config.num_eval_samples, 1, merged_config.infill_dataset, merged_config.vocab_file, model_exp_dir,0, 
    config=merged_config)

    print(f"return_best: {merged_config.return_best} return_best_logits: {merged_config.return_best_logits}\
           stability_coef: {merged_config.stability_coef} add_guidance: {merged_config.add_guidance}")
    print(f"Evaluted on {merged_config.num_eval_samples}, get solve rate: {solve_rate}")


if __name__ == "__main__":
    print("EVAL SEQ MODEL WARNING: must change in hydra.main the config file path to match saved model\n")
    eval()