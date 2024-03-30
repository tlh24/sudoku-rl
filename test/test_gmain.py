
import sys 
sys.path.insert(0,'/home/justin/sudoku-rl')
from gmain import generateActionValue, oneHotEncodeBoard, getDataLoaders, SudokuDataset, getDataDict
from sudoku_gen import Sudoku
from constants import *
from type_file import Action
import torch 
import os 

def test_generateActionValue():
   assert(generateActionValue(Action.DOWN.value, 1,1) == -1)
   assert(generateActionValue(Action.UP.value, 1,1) == 1)
   assert(generateActionValue(Action.LEFT.value, 1,1) == -1)
   assert(generateActionValue(Action.RIGHT.value, 1,1) == 1)
   assert(generateActionValue(Action.DOWN.value, 2,2) == -2)


def test_oneHotEncodeBoard():
   sudoku = Sudoku(SuN, SuK)
   curs_pos = torch.tensor([1,2])
   benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, Action.DOWN.value, -1)
   bvec = benc.squeeze(0)
   assert(bvec[0].item() == 1 and bvec[1].item() == 2 and all([bvec[i].item() == 0 for i in range(2, len(bvec))]))
   actvec = actenc.squeeze(0)
   assert (actvec[1].item() == -1 and all([actvec[i].item() == 0 for i in range(0, len(actvec)) if i != 1]))
   assert (newbenc[0,0] == 1)
   assert (newbenc[0,1] == 1)

   curs_pos = torch.tensor([1,2])
   benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, Action.UP.value, 2)
   actvec = actenc.squeeze(0)
   assert (actvec[1].item() == 2 and all([actvec[i].item() == 0 for i in range(0, len(actvec)) if i != 1]))
   assert (newbenc[0,0] == 1)
   assert (newbenc[0,1] == 4)

   curs_pos = torch.tensor([1,2])
   benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, Action.LEFT.value, -1)
   actvec = actenc.squeeze(0)
   assert (actvec[0].item() == -1 and all([actvec[i].item() == 0 for i in range(0, len(actvec)) if i != 0]))
   assert (newbenc[0,0] == 0)
   assert (newbenc[0,1] == 2)

def test_dataset():
   os.chdir('..')
   puzzles = torch.load('puzzles_500000.pt')
   NUM_SAMPLES = 12000
   NUM_EVAL = 2000
   data_dict = getDataDict(puzzles, NUM_SAMPLES, NUM_EVAL)
   train_dataset = SudokuDataset(data_dict['train_orig_board_encs'], data_dict['train_new_board_encs'],
                              data_dict['train_action_encs'], data_dict['train_graph_masks'],
                              data_dict['train_rewards'])
   
   batch = train_dataset[0]
   old_states = batch['orig_board']
   new_states = batch['new_board']
   actions = batch['action_enc']
   graph_masks = batch['graph_mask']
   rewards = batch['reward']
   pass 



def test_dataloaders():
   puzzles = torch.load('puzzles_500000.pt')
   NUM_SAMPLES = 12000
   NUM_EVAL = 2000
   train_dl, test_dl = getDataLoaders(puzzles, NUM_SAMPLES, NUM_EVAL)
   for batch in train_dl:
      old_states, new_states, actions, graph_masks, rewards = batch.values()
      assert(torch.equal(old_states[0] + actions[0], new_states[0]))
      
   



   


   

