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
sys.path.append(str(Path(__file__).parent.parent.absolute()))
print(sys.path)
from sudoku_gen import generateInitialBoard 
import time 
from collections import deque
from typing import List 
import copy 

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

puzzle = generateInitialBoard(0.7)
def isValidSudoku(board) -> bool:
    if isinstance(board, np.ndarray):
        board = board.tolist()
    
    for i in range(9):
        row = board[i]
        if len(row)!=len(set(row)): return False
        col = [board[c][i] for c in range(9)]
        if len(col)!=len(set(col)): return False
        box = [board[ind//3+(i//3)*3][ind%3+(i%3)*3] for ind in range(9)]
        if len(box)!=len(set(box)): return False
    return True



if __name__ == "__main__":
    new_board = generateInitialBoard(0.6)
    start_time = time.time()

    solver = SudokuSolver(new_board)
    if solver.solve():
        print("Sudoku solved:")
        solver.print_puzzle()
        print("\nTrajectory of board states:")
        #solver.print_trajectory()
    else:
        print("No solution exists")
    end_time = time.time()
    print(f"Solution took {end_time-start_time}s")

    print(f"Solution is valid: {isValidSudoku(solver.puzzle)}")


