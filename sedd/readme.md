## Background

This folder contains code to train score entropy discrete diffusion models for sudoku (among other things) datasets. 

For more information on score entropy discrete diffusion models, please refer to Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (Lou et al. 2023). 

This folder contains code derived from the original score entropy discrete diffusion github folder. Ref: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

The main learning problem in this folder is to learn the distribution of sudoku puzzles (for example SATNet or RRN puzzles). Practically, this means learning the distribution of 9x9 matrices which contain digits 1-9. Alternately, it is possible to instead learn trajectories of digit placements (for example, vectors of size < 81); this hasn't worked too well, but see data_utils.py and get_dataset() 'else' clause for sequence datasets.

This is an unsupervised learning problem (in that we do not require the initial board state but only final solutions). 
This is also a generative modeling problem. Discrete diffusion as compared to typical gaussian diffusion is natural because we are learning matrices that have discrete outputs (i.e digits 1-9). 

## Implementation notes 
- The input to train the SEDD model is the sudoku matrix solutions flattened into 1D vectors. Specifically, the input is a batch of vectors of 81 integers in [0,8], which correspond to digits 1-9.
- Our output matches the input distribution. Thus the output of SEDD model is a (batch) of vector of 81 integers in [0,8]. Note that when check whether a board is valid, we only check whether we have conflicts or not- we don't check if the digits are 1-9.  
- Under the hood, SEDD model is treating the input as indices (for example, instead of [0,8] corresponding to [1,9] they could correspond to nine letters). This is why our values must be 0 indexed. 
- Given a sequenece of integer indices, these are encoded into vectors through a learned embedding. 
- The SEDD model is composed of DDiT or denoising diffusion implicit transformer blocks which utilizes flash attention. The model architecture can be read in detail in model/transformer.py


## Training

To train, simply run ```python train.py```. Please first ensure that you have the correct conda environment activated. You may refer to ```sedd_env.yaml``` 

To modify the training run, please modify the config file under ```configs/normal_config.yaml```. 
In the config file, here are few things that may be relevant to modify.
- The dataset you are training on and validating (train/valid, not test) on— ```data: train``` and ```data: valid```
- number of training samples that you should train on— ```data: num_training_samples```
- The model configuration. You can increase the size of the model by changing ```model:n_blocks``` and ```model:n_heads```. If the sequence length is not 81, please change the sequence length ```model:length```

## Inference and evaluation 
The inference problem is $$f(x) \to y$$ where the model takes in an initial sudoku board $$x$$ and outputs a sudoku solution $$y$$. Since our diffusion model learns the distribution of completed solved puzzles $$y$$, to do inference we do conditional generation. At each diffusion timestep, for all given cells in the initial sudoku board (say with set of indices $$I$$), we replace the cells in the diffusion output with indices in $$I$$ with the correct initial cell in $$x$$. (i.e we paste the initial sudoku board onto the diffusion output at each timestep, leaving cell indices without a hint in $x$ unchanged.)

You can simply run ```python run_sampling.py``` and make sure to add the relevant arg flags, described below

- "--model_path" which is folder containing the saved model that should be evaluated 
- "--checkpoint_num" which is the checkpoint that specifies which saved model in the folder to use 
- "--dataset" which specififes which dataset to evaluate on 
- "--num_to_eval" which specifies the number of puzzles to evaluate
- "--steps" which specifices the number of reverse diffusion sampling steps. Typically, 256 or 512 is enough. In the actual paper, they go up to 2048, where they note that language generation is better with more diffusion timesteps.
- '--seq_len', which is the length of the data sequences we are trained on and generating.

## Code structure 
- ```configs/normal_config.yaml``` File that contains all configuration details. The other config files are just for reference
- ```experiments/``` Folder that will contain experiments and saved models
- ```model/``` Folder (copied from original sedd github) that contains model code. Most notably, see ```model/transformer.py```
-  ```catsample.py``` Utility file with categorical sampling helper functions
-  ```data_utils.py``` File that defines the datasets which can be used to train and evaluate the model. Also has dataloader code
-  ```graph_lib.py``` From sedd github, used for sedd
-  ```load_model.py``` From sedd github, utility file with config and model loading functions (also defines graph and noise)
-  ```losses.py``` From sedd github, utility file with loss helper functions
- ``` noise_lib.py``` From sedd github, utility file with noise helper functions
- ```run_sampling.py``` Main code to do model inference with saved model
- ```run_train.py``` Main code that defines model initialization/loading and training (and val) loop
- ```sampling_utils.py``` Utility file with helper functions that create initial board state (i.e partially complete puzzles) datasets for inference used in run_sampling.py and also evaluation statistics code
- ```sampling.py``` From sedd github, defines sedd sampling code and also includes conditional generation with infilling
- ```train.py``` Driver code to train your model
- ```utils.py``` General utility file with helper functions for logging, data preparation, and loss visualization 

Most important is ```train.py```, ```run_train.py```, ```run_sampling.py``` and ```data_utils.py```
