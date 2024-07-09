## Diffusion folder annotation

*main.py*

This contains driver code to train the diffusion model and also load the trained diffusion model and evaluate it on an antmaze environment. 
The trained diffusion model will be stored in a file. Note that the diffusion model is a wrapper that houses a temporalUnet model which can predict noise. 

*model.py*

Most importantly, this contains the class GaussianDiffusion and TemporalUnet. GaussianDiffusion is a wrapper that houses the temporalUnet noise prediction model.
It contains code to do a forward pass (to train the unet noise prediction model) and stores the noise scheduling constants. 

*utils.py*

This contains various util functions but most important contains the class Trainer. This loads a dataset and uses observation, action batches to 
train the model through the noise prediction error from noising the observation (note that the action is not used).

