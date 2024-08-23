from typing import List
from backtracking import isValidSudoku
import sys 
import numpy as np 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
print(sys.path)
from sudoku_gen import generateInitialBoard 
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

    def solve(self) -> bool:
        """
        Solve the Sudoku puzzle and store the trajectory of board states.

        :return: True if a solution is found, False otherwise.
        """
        self._build_positions_and_remaining()
        self._build_possibilities()
        numbers_to_fill = sorted(self.remaining, key=self.remaining.get)
        
        # Store initial state
        self.trajectory.append(("Initial", copy.deepcopy(self.puzzle)))
        
        result = self._fill_puzzle(0, numbers_to_fill, 0, list(self.possibilities[numbers_to_fill[0]].keys()))
        
        return result

    def _build_positions_and_remaining(self):
        """
        Build the positions and remaining dictionaries based on the initial puzzle state.
        """
        for i in range(9):
            for j in range(9):
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
        :param numbers: List of numbers to be filled in the puzzle.
        :param row_index: Index of the current row in the rows list.
        :param rows: List of rows where the current number can be placed.
        :return: True if a solution is found, False otherwise.
        """
        num = numbers[num_index]
        row = rows[row_index]

        for col in self.possibilities[num][row]:
            if self.puzzle[row][col] > 0:
                continue

            self.puzzle[row][col] = num
            self.trajectory.append((f"Place {num} at ({row}, {col})", copy.deepcopy(self.puzzle)))

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

        return False

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

solver = SudokuSolver(puzzle)


if solver.solve():
    print("Sudoku solved:")
    solver.print_puzzle()
    print("\nTrajectory of board states:")
    solver.print_trajectory()
else:
    print("No solution exists")