
import sys 
sys.path.insert(0,'/home/justin/sudoku-rl')
from gmain import generateActionValue, oneHotEncodeBoard
from sudoku_gen import Sudoku
from constants import *
from type_file import Action
import torch 

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
   assert (newbenc[0,0] == 1)
   assert (newbenc[0,1] == 1)

   curs_pos = torch.tensor([1,2])
   benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, Action.UP.value, 2)
   assert (newbenc[0,0] == 1)
   assert (newbenc[0,1] == 4)

   curs_pos = torch.tensor([1,2])
   benc, actenc, newbenc, msk, reward = oneHotEncodeBoard(sudoku, curs_pos, Action.LEFT.value, -1)
   assert (newbenc[0,0] == 0)
   assert (newbenc[0,1] == 2)

   


   

