'''
Train our nano-GPT model.
Boiler-plate code adapted from https://github.com/azreasoners/recurrent_transformer/blob/main/sudoku/main.py#L11
'''
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from datetime import datetime
import argparse 
from model import GPTConfig, GPT, Trainer, TrainerConfig 
import torch
import os 
from supervised.utils import set_seed, get_logger, restore_checkpoint
from guided_discrete.value.dataset import get_one_hot_dataset


def main(args=None):
    #NOTE: must supply a directory if evaluating to get checkpoint path 
    work_dir = 'results/09/22/2024:22:45:42'
    if not args.evaluate:
        work_dir = os.path.join(os.path.dirname(__file__), 'results', datetime.now().strftime("%m/%d/%Y:%H:%M:%S"))  
        os.makedirs(work_dir, exist_ok=True)

    checkpoint_path = os.path.join(work_dir, 'best.pth')
    ###
    #Setup
    ###
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = get_logger(os.path.join(work_dir, "logs"))
    logger.info(f"Working directory: {work_dir}")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"Found {torch.cuda.device_count()} CUDA devices.")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name}, Memory: {props.total_memory / (1024 ** 3):.2f}GB")
    logger.info(f"Number of CPUS: {os.cpu_count()}")

    ###
    #Load Data
    ###
    dataset = get_one_hot_dataset('rrn', True, 0.01, 0.1, is_value=True, num_samples=100000)
    indices = list(range(len(dataset)))

    test_dataset = torch.utils.data.Subset(dataset, indices[int(0.9 * len(dataset)):])
    val_dataset = torch.utils.data.Subset(dataset, indices[int(0.8 * len(dataset)):int(0.9 * len(dataset))])
    train_dataset = torch.utils.data.Subset(dataset, indices[:int(0.8 * len(dataset))])

    ###
    #Build GPT model and trainer
    ###
    model_conf = GPTConfig(vocab_size=9, block_size=81, n_head=args.n_head, n_embd=args.n_embd, num_classes=9,\
                            n_recur=args.n_recur, n_layer=args.n_layer)

    model = GPT(model_conf)
    logger.info(f"Configuration\n: {model_conf}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load in state
    state = dict(optimizer=optimizer, model=model, step=0) 
    
    if args.load_best:
        state = restore_checkpoint(checkpoint_path, state, device)
    
    initial_step = int(state['step'])

    tconf = TrainerConfig(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, val_interval = args.val_interval)
    trainer = Trainer(state, train_dataset, val_dataset, test_dataset, tconf, optimizer, logger, exp_dir=work_dir)
    logger.info(tconf)

    ###
    #Train model
    ###
    if not args.evaluate:
        print(f"Training model from step {initial_step}")
        trainer.train()
    
    
    ###
    #Evaluate
    ###
    # load the best checkpoint 
    model_conf = GPTConfig(vocab_size=9, block_size=81, n_head=args.n_head, n_embd=args.n_embd, num_classes=9,\
                            n_recur=args.n_recur, n_layer=args.n_layer)

    model = GPT(model_conf)
    state = dict(optimizer=optimizer, model=model, step=0) 
    state = restore_checkpoint(checkpoint_path, state, device)
    tconf = TrainerConfig(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, val_interval = args.val_interval)
    trainer = Trainer(state, train_dataset, val_dataset, test_dataset, tconf, optimizer, logger, exp_dir=work_dir)
    eval_loss = trainer.evaluate()
    print(f"Test loss: {eval_loss:.4f}")
    logger.info(f"Test loss: {eval_loss:.4f}")

    constraints_value_corr = trainer.evaluate_real()
    print(f"Correlation between the num of constraints and the value score: {constraints_value_corr:.4f}")
    logger.info(f"Correlation between the num of constraints and the value score: {constraints_value_corr:.4f}")
    return eval_loss
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--val_interval', type=int, default=1, help='Compute validation accuracy for how many number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')

    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--evaluate', action='store_true')

    parser.add_argument('--n_layer', type=int, default=1, help='Number of sequential self-attention blocks.')
    parser.add_argument('--n_recur', type=int, default=8, help='Number of recurrency of all self-attention blocks.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in each self-attention block.')
    parser.add_argument('--n_embd', type=int, default=128, help='Vector embedding size.')

    args = parser.parse_args()
    main(args)

