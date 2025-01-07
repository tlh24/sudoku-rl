## Background 
This folder contains code to train masked language model (MLM) discrete diffusion for sudoku (among other things) datasets.

This repo and method is adapted from, "Protein Design with Guided Discrete Diffusion" (Gruver et al. 2023).

This folder contains code derived from the github folder: [https://github.com/louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/ngruver/NOS)

The main learning problem in this folder is to learn the distribution of sudoku puzzles (for example SATNet or RRN puzzles). Practically, this means learning the distribution of 9x9 matrices which contain digits 1-9. This is an unsupervised learning problem (in that we do not require the initial board state but only final solutions). This is also a generative modeling problem. Discrete diffusion as compared to typical gaussian diffusion is natural because we are learning matrices that have discrete outputs (i.e digits 1-9).

The diffusion model of choice is Masked Language Model (MLM) variant. This means that the training objective is similar to BERT, predicting the unmasked tokens. Moreover this means that our model is using (as expected) a transformer backbone. For our forward process (noising process), we use the masked transition kernel, which means that with a certain probability $$\beta_t$$ the token in our sequence flips to [MASK] token and remains there. We schedule $$\beta_t$$ to be progressive increasing to 1 to ensure that over forward diffusion timesteps our sequence ends up becoming a sequence full of [MASK] tokens. 

## Implementation Notes
- The input to train the MLM model is the sudoku matrix solutions flattened into 1D vectors. Our masked langauge model input is a sequence of token indices, where these indices are defined based on the provided vocab.txt file. In the case of Sudoku, [MASK] refers to index 0 and digits "1"..."9" refer to indices 1-9. The initial data sequence is a batch of vectors of 81 integers in [0,8] corresponding to digits 1-9 which then become tokenized to a sequence of integers [1,9]. When the input sequence of token indices are corrupted, this will then include the [MASK] token or index 0, leading to a corrupted input sequence containing integers in [0,9]. For more information, refer to class SudokuLabeledDataset in data_utils.py
- Our output matches the input distribution. Thus the output of SEDD model is a (batch) of vector of 81 integers in [0,8]. Note that when check whether a board is valid, we only check whether we have conflicts or not- we don't check if the digits are 1-9.
- Under the hood, MLM model is treating the input as indices (for example, instead of [0,8] corresponding to [1,9] they could correspond to nine letters). This is why our values must be 0 indexed.
- Given a sequenece of integer indices, these are encoded into vectors through a learned embedding.
- The SEDD model is composed of DDiT or denoising diffusion implicit transformer blocks which utilizes flash attention. The model architecture can be read in detail in model/transformer.py


## Training model
To train the model please run python train_seq_model.py 
Note that you will have to change the configs file (such as train_seq_model.yaml or mlm.yaml) to customize training run


## Evaluating model 
- Change the eval ckpt in eval_seq_model.yaml
- Change the config path in eval_seq_model.py 
- 


