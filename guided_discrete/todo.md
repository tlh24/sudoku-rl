- [X] Find discrete diffusion repo 
- [X] Run an experiment with discrete diffusion (either clone or not)
- [X] Train a value function (predicts the number of constraints)
- [] 

Outstanding questions
- What is the guiding score? 
- What is get_labels 
-

# How to win today
- [X] Merge all code
- [X] Make sure that in checkpointing that we save model to the right path
- [X] Then load the right model 
- [X] Evaluate the trained MLM
- [X] Evaluate the trained value function
- [X] Add guidance
    - [] Prevent gradients updating the network and the value function 
    - [] Run experiments 
    - [X] Try different tau, try hard = True 

# Keep going 
- [] You need to merge the other branches like NOS into main to have the latest code  

- [] Add max over multiple sampling  


-[] Delete the option of outputting logit for [MASK] by altering CLS? 


