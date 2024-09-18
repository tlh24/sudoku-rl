import random 
import numpy as np 
import torch 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic=True


def getGitCommitHash():
	result = subprocess.run(
		['git', 'rev-parse', '--short', 'HEAD'],   # Command to run
		stdout=subprocess.PIPE,                    # Capture the output
		stderr=subprocess.PIPE,                    # Capture any errors
		text=True                                  # Return output as a string
	)
	if result.returncode == 0:
		return result.stdout.strip()  # Remove any trailing newlines
	else:
		raise RuntimeError(f"Error running git command: {result.stderr}")
