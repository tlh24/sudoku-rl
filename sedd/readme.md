## General notes 
- The SEDD model takes in a sequence of integers (token ids for example) and then 
these get encoded through a learned embedding 
- The input is indices; so this means that the values are 0-indexed. For a sudoku board of 81 cells, 
this could be for example a 81-length sequence of values in [0,8]
- Because this is diffusion, this means our output should also be in [0,8]. Note that when check whether a board is valid, we only check whether we have conflicts or not- we don't check if the digits are 1-9.


## Training new models
You can simply run python train.py
    To modify training, modify file (configs/normal_config.yaml)

## Run sampling
You can simply run python run_sampling.py and make sure to add the relevant arg flags 


