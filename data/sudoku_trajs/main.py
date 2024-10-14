'''
Generate backtracking trajectories and also labels corresponding to being backtracked or not 

- create backtracked state trajectories
    - label backtracked trajectories based on going forward or backwards
- convert the state trajectories to action sequences  
'''
import os 
import sys 
import numpy as np 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
print(sys.path)
from sudoku_gen import generateInitialBoard 
import time 
from collections import deque
from typing import List 
import copy 
from utils import actionTupleToAction
import logging
import time 

class SudokuSolver:
    """
    A class to solve Sudoku puzzles using a backtracking algorithm,
    with the ability to track the trajectory of board states.
    """

    def __init__(self, puzzle: List[List[int]]):
        """
        Initialize the SudokuSolver with a given puzzle.

        :param puzzle: A 9x9 list of lists representing the Sudoku puzzle.
                       0 represents an empty cell.
        """
        self.puzzle = puzzle
        self.positions = {}  # {number: [positions]}
        self.remaining = {}  # {number: count}
        self.possibilities = {}  # {number: {row: [possible_columns]}}
        self.trajectory = []  # List to store board states
        self.not_backtracked_state = [] # True if the state did not result from having a digit removed, False if state results from having digit removed

    def solve(self) -> bool:
        """
        Solve the Sudoku puzzle and store the trajectory of board states.

        :return: True if a solution is found, False otherwise.
        """
        self._build_positions_and_remaining()
        self._build_possibilities()
        # sort the digits to place by how completed they are, i.e how many of it has already been placed
        sorted_remaining = {k:v for k,v in sorted(self.remaining.items(), key=lambda x: x[1])} 
        # get all digits that haven't been placed 9 times in order of how completed it is
        numbers_to_fill = [num for num in sorted_remaining.keys() if sorted_remaining[num] > 0]

        # Store initial state
        self.trajectory.append(("Initial", copy.deepcopy(self.puzzle)))
        self.not_backtracked_state.append(True)

        assert len(list(self.possibilities[numbers_to_fill[0]].keys())) > 0
        
        result = self._fill_puzzle(0, numbers_to_fill, 0, list(self.possibilities[numbers_to_fill[0]].keys()))
        
        return result

    def _build_positions_and_remaining(self):
        """
        Build the positions and remaining dictionaries based on the initial puzzle state.
            Sets any digit values (1-9) that don't exist to have [] in positions and 9 in remaining 
        """
        for i in range(9):
            for j in range(9):
                # for all digit values, store the location in positions and decrement the number of digits (with the same val) remaining to place 
                num = self.puzzle[i][j]
                if num > 0:
                    self.positions.setdefault(num, []).append([i, j])
                    self.remaining[num] = self.remaining.get(num, 9) - 1

        for num in range(1, 10):
            if num not in self.positions:
                self.positions[num] = []
            if num not in self.remaining:
                self.remaining[num] = 9

    def _build_possibilities(self):
        """
        Build the possibilities dictionary, which stores possible positions for each number.
        Possibilities dictionary is {number: {row: [possible_columns]}}
        """
        for num, positions in self.positions.items():
            self.possibilities[num] = {}
            available_rows = set(range(9))
            available_cols = set(range(9))

            for row, col in positions:
                available_rows.discard(row)
                available_cols.discard(col)

            for row in available_rows:
                possible_cols = [col for col in available_cols if self.puzzle[row][col] == 0]
                if possible_cols:
                    self.possibilities[num][row] = possible_cols

    def _is_safe(self, row: int, col: int) -> bool:
        """
        Check if it's safe to place a number in the given position.

        :param row: Row index of the position to check.
        :param col: Column index of the position to check.
        :return: True if it's safe to place the number, False otherwise.
        """
        num = self.puzzle[row][col]
        
        # Check row and column
        for i in range(9):
            if i != col and self.puzzle[row][i] == num:
                return False
            if i != row and self.puzzle[i][col] == num:
                return False

        # Check 3x3 sub-box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if i != row and j != col and self.puzzle[i][j] == num:
                    return False

        return True

    def _fill_puzzle(self, num_index: int, numbers: List[int], row_index: int, rows: List[int]) -> bool:
        """
        Recursively fill the puzzle using backtracking, storing each state change.

        :param num_index: Index of the current number in the numbers list.
        :param numbers: List of numbers to be filled in the puzzle (1-9).
        :param row_index: Index of the current row in the rows list.
        :param rows: List of rows where the current number can be placed.
        :return: True if a solution is found, False otherwise.
        """
        num = numbers[num_index]
        row = rows[row_index]
        # try placing the num for all possible cols corresponding to a given row
        for col in self.possibilities[num][row]:
            if self.puzzle[row][col] > 0:
                continue

            self.puzzle[row][col] = num
            self.trajectory.append((f"Place {num} at ({row}, {col})", copy.deepcopy(self.puzzle)))
            self.not_backtracked_state.append(True)

            if self._is_safe(row, col):
                if row_index < len(rows) - 1:
                    if self._fill_puzzle(num_index, numbers, row_index + 1, rows):
                        return True
                elif num_index < len(numbers) - 1:
                    next_num = numbers[num_index + 1]
                    if self._fill_puzzle(num_index + 1, numbers, 0, list(self.possibilities[next_num].keys())):
                        return True
                else:
                    return True

            # Backtrack
            self.puzzle[row][col] = 0
            self.trajectory.append((f"Remove {num} from ({row}, {col}) - Backtrack", copy.deepcopy(self.puzzle)))
            self.not_backtracked_state.append(False)

        return False
    
    def get_forward_trajectory(self):
        '''
        From the trajectory including backtracked states, only return a list of forward states
            ex: S1 S2 B S3 -> S1 S3
            ex: S1 S2 S3 B B S4 -> S1 S4
        '''
        self.forward_states = deque()
        for i in range(0, len(self.trajectory)):
            if self.not_backtracked_state[i]:
                self.forward_states.append(self.trajectory[i])
            else:
                self.forward_states.pop()
                
        return self.forward_states

    def print_trajectory(self):
        """
        Print the trajectory of board states.
        """
        print(len(self.trajectory))
        for i, (action, state) in enumerate(self.trajectory):
            print(f"Step {i}: {action}")
            self._print_state(state)
            print()

    def _print_state(self, state):
        """
        Print a single board state.

        :param state: The board state to print.
        """
        for row in state:
            print(' '.join(str(num) if num != 0 else '.' for num in row))
    
    def print_puzzle(self):
        """
        Print the current state of the puzzle in a readable format.
        """
        for row in self.puzzle:
            print(' '.join(str(num) if num != 0 else '.' for num in row))


class TrajectorySaver:
    def __init__(self, is_backtracking=False, is_action=False):
        self.is_backtracking = is_backtracking
        self.is_action = is_action 
        if self.is_backtracking:
            if self.is_action:
                self.save_folder = os.path.join('backtracking/action/') 
            else:
                self.save_folder = os.path.join('backtracking/state') 
        else:
            if self.is_action:
                self.save_folder = os.path.join('forward/action')
            else: 
                self.save_folder = os.path.join('forward/state')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def generate(self, num_trajs: int, percent_filled:float=0.5):
        #Saves trajecotories of states, where start from initial puzzle and go to end solution
        # Generation doesn't work for percent less than 0.4 
        save_path = os.path.join(self.save_folder, f'{num_trajs}_trajs_{percent_filled}_filled_no_head.npy')
        new_board = generateInitialBoard(percent_filled, True)
        solver = SudokuSolver(new_board)
        solver.solve()
        example_traj = solver.get_forward_trajectory()
        
        trajs = np.zeros((num_trajs, len(example_traj), 9, 9)) 
        for i in range(num_trajs):
            new_board = generateInitialBoard(percent_filled)
            solver = SudokuSolver(new_board)
            if solver.solve():
                # TODO: currently only saves forward states 
                annotated_forward_traj = solver.get_forward_trajectory()
                forward_traj = [tuple[1] for tuple in annotated_forward_traj]
                trajs[i] = np.array(forward_traj)
        
        np.save(save_path, trajs)
    
    def insert_begin_states(self, state_trajs_file: str):
        '''
        Given a file that contain state trajectories, prepends a sequence of states starting from empty board that leads to the
        beginning state traj
        Saves new traj file with head 

        state_trajs_file: (str) path to .npy
        '''
        
        state_trajs = np.load(state_trajs_file) #(num_samples, traj_length, 9,9)
        num_samples = state_trajs.shape[0]
        new_trajs = [] 
        for sample_idx in range(0, num_samples):
            orig_traj = state_trajs[sample_idx]
            first_state = orig_traj[0]
            head_states = []
            # add the digits one by one going from left->right, top->down
            current_state = np.zeros((9,9))
            head_states.append(current_state)
            for i in range(0, 9):
                for j in range(0, 9):
                    if first_state[i][j] > 0:
                        current_state = copy.deepcopy(current_state)
                        current_state[i][j] = first_state[i][j]
                        head_states.append(current_state)
            
            head_states = np.array(head_states[:-1])
            
            assert orig_traj[0].shape == head_states[0].shape 
            new_traj = np.concatenate((head_states, orig_traj))
            new_trajs.append(new_traj)
        
        new_trajs = np.array(new_trajs)
        new_file_name = state_trajs_file.replace("no", "yes")
        np.save(new_file_name, new_trajs)
        
    
    def save_action_trajs_from_state_trajs(self, state_trajs_file: str):
        '''
        Convert a trajectory of n states into a trajectory of n-1 actions (numbers in [0, 728]).
            Note: for this to be well-defined, state_trajs_file must have "yes" head, i.e contains the states that lead to the initial puzzle  
        '''
        assert "yes" in state_trajs_file
        state_trajs = np.load(state_trajs_file) #(num_samples, traj_length, 9,9)
        num_samples = state_trajs.shape[0]
        new_trajs = [] 
        for i in range(0, num_samples):
            new_traj = []
            traj = state_trajs[i]
            for t in range(0, len(traj)-1):
                difference = traj[t+1]-traj[t]
                non_zero_idxs = np.nonzero(difference)
                try: 
                    assert len(non_zero_idxs[0]) == 1
                except AssertionError:
                    logging.error(f"Difference matrix {difference} has more than one difference")
                i,j = non_zero_idxs[0][0], non_zero_idxs[1][0]
                digit = int(difference[i][j]) 
                assert digit > 0 
                action = actionTupleToAction((i,j,digit))
                new_traj.append(action)

            new_trajs.append(new_traj)
        save_trajs = np.array(new_trajs)
        new_file_path = state_trajs_file.replace("state", "action")
        np.save(new_file_path, save_trajs)




  

if __name__ == "__main__":
    #NOTE: For relatives path to work, must execute this in sudoku-rl/data/sudoku_trajs 
    assert 'data/sudoku_trajs' in os.getcwd()

    #saver = TrajectorySaver()
    #saver.generate(10000, 0.45)
    #saver.insert_begin_states('forward/state/10000_trajs_0.45_filled_no_head.npy')
    #saver.save_action_trajs_from_state_trajs('forward/state/10000_trajs_0.45_filled_yes_head.npy')
    hard_puzzle = [
        [0, 6, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 3, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 2, 4],
        [8, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 7, 5, 0],
        [2, 0, 0, 9, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 6, 0, 0]
    ]
    print("Initial puzzle is \n")
    solver = SudokuSolver(hard_puzzle)
    solver.print_puzzle()
    start_time = time.time()
    result = solver.solve()
    print("Solution is \n")
    solver.print_puzzle()
    end_time = time.time()
    print(f"Total time taken is {end_time - start_time} sec")


