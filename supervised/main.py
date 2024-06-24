'''
Train our nano-GPT model.
Boiler-plate code adapted from https://github.com/azreasoners/recurrent_transformer/blob/main/sudoku/main.py#L11
'''
import argparse 
from utils import set_seed, Sudoku_Dataset_SATNet, GPTConfig, GPT, Trainer, TrainerConfig 
import torch


def main(args=None):
    set_seed(42)

    ###
    #Load Data
    ###
    dataset = Sudoku_Dataset_SATNet()
    indices = list(range(len(dataset)))

    test_dataset = torch.utils.data.Subset(dataset, indices[-1000:])
    val_dataset = torch.utils.data.Subset(dataset, indices[8000:-1000])
    train_dataset = torch.utils.data.Subset(dataset, indices[:8000])

    ###
    #Build GPT model and trainer
    ###
    model_conf = GPTConfig(vocab_size=10, block_size=81, n_head=args.n_head, n_embd=args.n_embd, num_classes=9,\
                            n_recur=args.n_recur, n_layer=args.n_layer)

    model = GPT(model_conf)

    tconf = TrainerConfig(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, val_interval = args.val_interval)
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, tconf)

    ###
    #Train model
    ###

    trainer.train()
    eval_acc = trainer.evaluate()
    print(f"Test Acc: {eval_acc:.4f}")
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--val_interval', type=int, default=10, help='Compute validation accuracy for how many number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')

    parser.add_argument('--n_layer', type=int, default=1, help='Number of sequential self-attention blocks.')
    parser.add_argument('--n_recur', type=int, default=32, help='Number of recurrency of all self-attention blocks.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of heads in each self-attention block.')
    parser.add_argument('--n_embd', type=int, default=128, help='Vector embedding size.')

    args = parser.parse_args()
    main(args)

