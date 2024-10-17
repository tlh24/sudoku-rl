from torch.utils.data import Dataset
import torch 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
data_dir = os.path.join(project_dir, 'data')

class Sudoku_SATNet(Dataset):
    def __init__(self):
        data = {}
        data_to_path = {
            'board': os.path.join(data_dir, 'satnet', 'features.pt'),
            'board_img': os.path.join(data_dir, 'satnet', 'features_img.pt'),
            'label': os.path.join(data_dir, 'satnet', 'labels.pt'),
            'perm': os.path.join(data_dir, 'satnet', 'perm.pt'),
        }
        for k in data_to_path:
            with open(data_to_path[k], 'rb') as f:
                data[k] = torch.load(f)
        # board has shape (10000, 81), 0's with no digits and 1-9 for digits
        self.board = ((data['board'].sum(-1) != 0) * (data['board'].argmax(-1) + 1)).view(-1, 81).long() 
        self.label = data['label'].argmax(-1).view(-1, 81).long() # (10000, 81)
        self.label_ug = self.label.clone() # (10000, 81)
        # label_ug is a label vector of indices(0-8 vs 1-9) of size 81 but all initially given digits are -100
        self.label_ug[self.board != 0] = -100

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        """
        Returns label: which is a float tensor of shape (81) consisting of {0,...,8}
        """
        label = self.label[idx] #convert solution to have digits in [0,8]
        return label 


if __name__ == "__main__":
    pass 