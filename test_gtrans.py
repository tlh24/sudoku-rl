import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pdb
import matplotlib.pyplot as plt
from gracoonizer import Gracoonizer
from constants import *
from tqdm import tqdm
import psgd


class SimpleMLP(nn.Module):
    def __init__(self, enc_dim):
        super().__init__()
        self.input_fc = nn.Linear(enc_dim*2, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, enc_dim)

    def forward(self, old_states, actions, masks, epoch, _):
        # old_states shape [batch size, 1, enc_dim]
        # actions shape [batch size, 1, enc_dim]

        batch_size = old_states.shape[0]
        # combine state and action and flatten
        x = torch.cat((old_states, actions), 1).view(batch_size, -1)

        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        # y shape [batch size, 1, output_dim]
        y = self.output_fc(x).unsqueeze(1)

        return y, None, None, None, None, None


class BinaryDataset(Dataset):
    '''
    Dataset where the ith element is a sample containing random vector,
            zeros/ones vector, zero/one int, graph_mask, board_reward
  If int is zero, needs to predict zeros vector; else ones vector
    '''

    def __init__(self, num_samples, enc_dim=20):
        self.orig_boards_enc = torch.randint(
            low=0, high=10, size=(num_samples, enc_dim)).unsqueeze(1).float()
        actions = torch.randint(low=0, high=2, size=(num_samples, 1))
        new_state = actions.expand(num_samples, enc_dim).unsqueeze(1).float()
      # zero pad action enc to be length enc_dim
        actions_enc = F.pad(actions, (0, enc_dim-1),
                            "constant", 0).unsqueeze(1).float()
        self.actions_enc = actions_enc
        self.new_boards_enc = new_state
        self.graph_masks = torch.ones((num_samples, 2, 2))
        self.rewards = torch.zeros(num_samples)

    def __len__(self):
        return self.orig_boards_enc.size(0)

    def __getitem__(self, idx):
        sample = {
            'orig_board': self.orig_boards_enc[idx],
            'new_board': self.new_boards_enc[idx],
            'action_enc': self.actions_enc[idx],
            'graph_mask': self.graph_masks[idx],
            'reward': self.rewards[idx].item(),
        }
        return sample


def getTestDataLoaders(num_samples, num_eval=2000):
    '''
    Returns a train and test dataloader for a mock dataset what has 
    an 2d vector, a binary number (0 or 1), and corresponding vector of 
zeros or ones
    '''
    train_dataset = BinaryDataset(num_samples-num_eval)
    test_dataset = BinaryDataset(num_eval)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def train(args, memory_dict, model, train_loader, optimizer, criterion, epoch):
    model.train()
    sum_batch_loss = 0.0

    for batch_idx, batch_data in enumerate(train_loader):
        old_states, new_states, actions, graph_masks, rewards = [
            t.to(args["device"]) for t in batch_data.values()]

        pred_data = {}
        if optimizer_name != 'psgd':
            optimizer.zero_grad()
            new_state_preds, reward_preds, a1, a2, w1, w2 = model.forward(
                old_states, actions, None, epoch, None)
            pred_data = {'old_states': old_states, 'new_states': new_states, 'new_state_preds': new_state_preds,
                         'rewards': rewards, 'reward_preds': reward_preds,
                         'a1': a1, 'a2': a2, 'w1': w1, 'w2': w2}
            loss = criterion(new_state_preds, new_states)
            loss.backward()
            optimizer.step()
            # print(loss.detach().cpu().item())
        else:
            # psgd library already does loss backwards and zero grad
            def closure():
                nonlocal pred_data
                new_state_preds, reward_preds, a1, a2, w1, w2 = model.forward(
                    old_states, actions, None, epoch, None)
                pred_data = {'old_states': old_states, 'new_states': new_states, 'new_state_preds': new_state_preds,
                             'rewards': rewards, 'reward_preds': reward_preds,
                             'a1': a1, 'a2': a2, 'w1': w1, 'w2': w2}
                loss = torch.sum((new_state_preds - new_states)**2) + sum(
                    [torch.sum(1e-4 * torch.rand_like(param) * param * param) for param in model.parameters()])
                return loss
            loss = optimizer.step(closure)
            # print(loss.detach().cpu().item())

        sum_batch_loss += loss.cpu().item()

    # add epoch loss
    avg_batch_loss = sum_batch_loss / len(train_loader)
    args["fd_losslog"].write(f'{epoch}\t{avg_batch_loss}\n')
    args["fd_losslog"].flush()


def getOptimizer(optimizer_name, model, lr=1e-3, weight_decay=0):
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    else:
        optimizer = psgd.LRA(model.parameters(), lr_params=0.01, lr_preconditioner=0.01, momentum=0.9,
                             preconditioner_update_probability=0.1, exact_hessian_vector_product=False, rank_of_approximation=10, grad_clip_max_norm=5)
    return optimizer


if __name__ == '__main__':
    NUM_SAMPLES = 12000
    NUM_EVAL = 2000
    NUM_EPOCHS = 500
    device = torch.device('cuda:0')
    fd_losslog = open('testlosslog.txt', 'w')
    args = {"NUM_SAMPLES": NUM_SAMPLES, "NUM_EPOCHS": NUM_EPOCHS, "NUM_EVAL": NUM_EVAL,
            "device": device, "fd_losslog": fd_losslog}
    optimizer_name = "adamw"

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(42)

    # get our train and test dataloaders
    train_dataloader, test_dataloader = getTestDataLoaders(10000)

    model = SimpleMLP(20).to(device)

    optimizer = getOptimizer(optimizer_name, model)
    criterion = nn.MSELoss()

    epoch_num = 0
    for _ in tqdm(range(0, args["NUM_EPOCHS"])):
        train(args, None, model, train_dataloader,
              optimizer, criterion, epoch_num)
        epoch_num += 1

    # save after training
    model.save_checkpoint()

    print("validation")
    # validate(args, model, test_dataloader, criterion, epoch_num)
