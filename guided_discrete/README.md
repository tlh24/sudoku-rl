## Background 
This folder contains code to train masked language model (MLM) discrete diffusion for sudoku (among other things) datasets.
This repo and method is adapted from, "Protein Design with Guided Discrete Diffusion" (Gruver et al. 2023).
This folder contains code derived from the github folder: [https://github.com/louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/ngruver/NOS)

Note that this folder uses pytorch lightning and so the training and evaluation loop is defined with the model. 

The main learning problem in this folder is to learn the distribution of sudoku puzzles (for example SATNet or RRN puzzles). Practically, this means learning the distribution of 9x9 matrices which contain digits 1-9. This is an unsupervised learning problem (in that we do not require the initial board state but only final solutions). This is also a generative modeling problem. Discrete diffusion as compared to typical gaussian diffusion is natural because we are learning over discrete outputs (i.e digits 1-9).

The diffusion model of choice is Masked Language Model (MLM) variant. This means that the training objective is similar to BERT, predicting the unmasked tokens and using a likelihood objective. Moreover this means that our model is using (as expected) a transformer backbone. For our forward process (noising process), we use the masked transition kernel, which means that with a certain probability $$\beta_t$$ the token in our sequence flips to [MASK] token and remains there. We schedule $$\beta_t$$ to be progressive increasing to 1 to ensure that over forward diffusion timesteps our sequence ends up becoming a sequence full of [MASK] tokens. 

## Implementation Notes
- The input to train the MLM model is a sequence of token indices corresponding to the sudoku matrix solutions flattened into 1D vectors. These token defined based on the provided vocab.txt file. In the case of sudoku_vocab.txt, [MASK] refers to index 0 and digits "1"..."9" refer to indices 1-9. Initially, the data is batch of vectors of 81 integers in [0,8] corresponding to digits 1-9—but this becomes tokenized to a sequence of integers [1,9] following the sudoku_vocab.txt. When the input sequence of token indices are corrupted, this will then include the [MASK] token or index 0, leading to a corrupted input sequence containing integers in [0,9]. For more information, refer to class SudokuLabeledDataset() in ```data_utils.py```
- Our output matches the input distribution. Thus the output of model is a (batch) of vector of 81 integers in hopefully [1,9], but technically in [0,9]; if [MASK] token is included then learning has gone awry.

## Setup 
The conda environment yaml ```environment.yml``` is included for easy installation. Please activate the environment before runnning these scripts. 

## Training model
To train the model please run ```python train_seq_model.py```. 
To customize the training run you will need to change the config (such as ```train_seq_model.yaml``` for general training settings or ```mlm.yaml``` for model specific settings).

In the ```train_seq_model.yaml``` file, here are few things that may be relevant to modify.
- The dataset you are training on and validating (train/valid, not test) on— ```train``` and ```valid```
- the dataset that you are doing conditional generation for (ex: to get some sense on how well it infills and solves a partial sudoku puzzle) ```infill_dataset```
- Boolean whether training or evaluation— ```is_eval```

In the ```mlm_model.yaml``` file, here are few things that may be relevant to modify to toggle the model size

- Dropout in model: ```dropout```
- Bert config name to load the model: ```bert_config_name```
- Number of hidden layers, can override default bert config:  ```num_hidden_layers```
- Number of attention heads, can override default bert config: ```num_attention_heads```
- Number of diffusion timesteps: ```noise_schedule: timesteps```
- Optimizer lr: ```optimizer: lr``` **Note:** From personal experience, if the *lr* is too large then training will blow up. There needs to be hyperparameter search with *lr* to ensure good performance.


## Evaluating model 
To evaluate the model, please first change the relevant parts of ```eval_seq_model.yaml```. Specifically, ensure to
- Change the eval_ckpt to decide which model to evaluate
- Change the config path in the hydra.main() in ```eval_seq_model.py``` to match the eval_ckpt

Then you can simply run ```python eval_seq_model.py```

## Code Structure 
- ```configs/train_seq_model.yaml``` Training configuration settings
- ```configs/eval_seq_model.yaml``` Evaluation configuration settings
- ```configs/model/mlm.yaml``` Model configuration settings
- ```seq_model``` Folder contains code that defines the model. Most important is ```mlm_diffusion.py``` which defines the MLM model and also the relevant training and inference code (such as guidance during inference time).
- ```data_utils.py``` Util file to create datasets
-  ```eval_seq_model.py``` Driver code to evaluate a particular model
-  ```sample.py``` Utility script contains function to generate condition samples and evaluate them, as used in eval_seq_model 
- ```train_seq_model.py``` Driver code to train the model
- ```trainer.py``` File contains pytorch lightning code which defines training loop and also validation and logging

## Better understanding Guided Discrete Diffusion from the original Gruver paper
- Most helpful would be to first read through the poster which explains the overall method and guidance method. Also it would be helpful to read the reference Gruver paper. However note that the Gruver paper does not match what their github contains. Adding the email thread with Gruver below 
- In the code, you may note that in ```data_utils.py``` our dataset contains not just the token sequence but also a *corrupt_mask* and *attn_mask*. These are initialized to all ones vector. For our purpose, *attn_mask* remains as one vector and so effectively can be ignored. However the *corrupt_mask* defines which tokens should be corrupted (based on some flipping of a coin of probability $$\beta_t$$), see ```noise_schedule.py```. 
In our mask likelihood objective, we only care about the likelihood of tokens that were corrupted, so we can use the corrupt mask for this purpose.


#### Email thread clarifying Gruver code
> One question, I am having trouble seeing the Langevin dynamics sample step in the code for categorical diffusion.
<img src="https://github.com/user-attachments/assets/b457b7de-778f-4c65-88f5-cbcad11d7a7c" width="500"/>

> Specifically, I cannot identify where \(\sum_{\hat{w}} p(w_{t-1} | w_t, \hat{w}) p_\theta (\hat{w} | w_t)\) is calculated to be used in the KL divergence loss.  
Looking at the `guidance_steps()` function in `mlm_diffusion.py`, it seems that the KL divergence is calculating with respect to the logits that is outputted from the denoiser model (in this case `self.network`), \(f(x_t)\) based on some \(x_t\) that is a noised sample coming from the previous logits.

> <img src="https://github.com/user-attachments/assets/0d6d8857-51d2-4115-b917-ebc78258d391" width="500"/>


His response
> Great question! You're right that there's a slight mismatch between the appendix algorithm and the implementation here. Also notably I'm using Adagrad which has some higher order terms that aren't in the appendix algorithm because I found that in practice it allowed me to take less gradient steps on the value function :). Re: using the KL on p(w_{t-1} | w_t)) vs p(\hat{w}|w_t), I believe it should result in something roughly similar but with a slightly different weighting over the tokens. I suspect I tried it and found that it worked slightly worse or similar, but you could try it out! Note that I'm also using the unweighted loss here: https://github.com/ngruver/NOS/blob/main/seq_models/model/mlm_diffusion.py#L177. I believe it's common to use an unweighted loss when the priority is sample quality and weighted losses when the priority is good likelihoods/perplexity.


