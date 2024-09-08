import torch
from torch.utils.data import Dataset

DATA_PATH = "./data/satnet"


class SudokuSATNetDataset(Dataset):
    def __init__(self):
        data = {}
        data_to_path = {
            "board": f"{DATA_PATH}/features.pt",
            "board_img": f"{DATA_PATH}/features_img.pt",
            "label": f"{DATA_PATH}/labels.pt",
            "perm": f"{DATA_PATH}/perm.pt",
        }
        for k in data_to_path:
            with open(data_to_path[k], "rb") as f:
                data[k] = torch.load(f)
        self.board = (
            ((data["board"].sum(-1) != 0) * (data["board"].argmax(-1) + 1))
            .view(-1, 81)
            .long()
        )  # (10000, 81)
        # self.board_img = data['board_img'].view(10000, 81, 28, 28).float() # (10000, 81, 28, 28)
        self.label = data["label"].argmax(-1).view(-1, 81).long()  # (10000, 81)
        self.label_ug = self.label.clone()  # (10000, 81)
        # label_ug is a label vector of indices(0-8 vs 1-9) of size 81 but all initially given digits are -100
        self.label_ug[self.board != 0] = -100

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Each data instance is a tuple <board, board_img, label, label_ug> where
            board: a float tensor of shape (81) consisting of {0,...,9}
            board_img: a float tensor of shape (81, 28, 28) denoting 81 MNIST images
            label: a float tensor of shape (81) consisting of {0,...,8}
            label_ug: a float tensor of shape (81) consisting of {0,...,8} and -100 denoting given cells
        Note:
            We only use the pair <board, label_ug> as a data instance for textual Sudoku
        """
        # return self.board[idx], self.board_img[idx], self.label[idx], self.label_ug[idx]
        return self.board[idx], self.label_ug[idx]
