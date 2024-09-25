import datetime
import os
import os.path
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data_utils as data 
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from utils import action_seq_to_board, action_traj_idxs_unique
torch.backends.cudnn.benchmark = True
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from utils import isValidSudoku

def _run(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    work_dir = cfg['work_dir']

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)


    # logging
    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Configuration: {cfg}")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"Found {torch.cuda.device_count()} CUDA devices.")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}, Memory: {props.total_memory / (1024 ** 3):.2f}GB")
    logger.info(f"Number of CPUS: {os.cpu_count()}")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    num_parameters = sum(p.numel() for p in score_model.parameters())
    logger.info(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg['training']['ema'])
    logger.info(f"Model: {score_model}")
    logger.info(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    logger.info(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    logger.info(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # load in tokenizer
    if cfg['data']['is_hugging']:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg['training']['accum'])
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg['training']['accum'])


    if cfg['training']['snapshot_sampling']:
        sampling_shape = (cfg['training']['batch_size'], cfg['model']['length'])
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg['training']['n_iters']
    logger.info(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']
        if cfg['data']['is_hugging']:
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)
        
        loss = train_step_fn(state, batch)
        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            # logging
            if step % cfg['training']['log_freq'] == 0:
                logger.info(f"step: {step}, training_loss: {loss.item():.5e}")
            # checkpointing 
            if step % cfg['training']['snapshot_freq_for_preemption'] == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            # evaluation 
            if step % cfg['training']['eval_freq'] == 0:
                if cfg['data']['is_hugging']:
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                else:
                    eval_batch = next(eval_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)
                logger.info(f"step: {step}, evaluation_loss: {eval_loss.item():.5e}")
            
            # Sampling and saving
            if step > 0 and step % cfg['training']['snapshot_freq'] == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg['training']['snapshot_freq']
                utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                if cfg['training']['snapshot_sampling']:
                    logger.info(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, f"iter_{step}")
                    os.makedirs(this_sample_dir, exist_ok=True)
                    file_name = os.path.join(this_sample_dir, "sample.txt")

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())
                    if cfg['data']['is_hugging']:
                        sentences = tokenizer.batch_decode(sample)
                        with open(file_name, 'w') as file:
                            for sentence in sentences:
                                file.write(sentence + "\n")
                                file.write("============================================================================================\n")

                        if cfg['eval']['perplexity']:
                            with torch.no_grad():
                                eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                                batches = sample.shape[0] // cfg['eval']['perplexity_batch_size']
                                total_perplexity = 0
                                for i in range(batches):
                                    s = sample[i * cfg['eval']['perplexity_batch_size']:(i + 1) * cfg['eval']['perplexity_batch_size']]
                                    loss, logits = eval_model(s, labels=s)[:2]
                                    logits = logits.transpose(-1, -2)
                                    perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                    total_perplexity += perplexity
                                total_perplexity /= batches
                                logger.info(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")

                                del eval_model, logits, loss

                    else:
                        output_boards = []
                        for i in range(len(sample)):
                            seq = sample[i].cpu().detach().numpy()
                            #action_traj_idxs_unique(seq)
                            assert seq.shape == (81,) or seq.shape == (1,81)
                            digit_seq = seq + 1
                            output_boards.append(digit_seq.reshape((9,9)))
                            #output_boards.append(action_seq_to_board(seq))

                        valid_results = [] 
                        
                        with open(file_name, 'w') as file:
                            for board in output_boards:
                                for row in board:
                                    row = [int(num) for num in row]
                                    row_str = ' '.join(map(str, row))
                                    file.write(row_str + "\n")
                            
                                is_valid = isValidSudoku(board)
                                valid_results.append(is_valid)
                                file.write(f"Is valid: {is_valid}\n")
                                file.write("============================================================================================\n")

                            #TODO: add eval step
                            correct_vals = [x for x in valid_results if x]
                            acc = len(correct_vals)/len(valid_results)
                            file.write(f"Overall accuracy: {len(correct_vals)}/{len(valid_results)} = {acc:.2f}")

                        print(f"Overall accuracy: {len(correct_vals)}/{len(valid_results)} = {acc:.2f}")
        else:
            raise ValueError("Model step has not changed")

