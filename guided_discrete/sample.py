import os 
import torch 
import copy 
import transformers 
import sys 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from sedd.sampling_utils import get_test_puzzles, evaluate_samples


def test_solving(
    model,
    num_samples,
    infill_dataset: str, 
    vocab_file,
    exp_dir, 
    guidance_kwargs=None
):
    '''
    Generate conditional samples and test solution accuracy 
    '''
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=vocab_file,
        do_lower_case=False 
    )
    device = next(model.parameters()).device 
    puzzles = get_test_puzzles(infill_dataset, num_samples, device) #tensor of shape (num_samples, 81) values 0-8, -1 for incomplete
    for i, puzzle in enumerate(puzzles):
        list_digits = puzzle.tolist()
        list_chars = list(map(str, list_digits))
        # replace all empty cells (indicated by negative values) to MASK
        for i in range(0, len(list_chars)):
            if list_digits[i] < 0:
                list_chars[i] = '[MASK]'
        infill_seed = torch.Tensor(
            tokenizer.convert_tokens_to_ids(list_chars)
        ).long().to(device)
        
        infill_mask = infill_seed == tokenizer.mask_token_id
        assert torch.sum(infill_mask.long()).detach().cpu() > 10
        #corrupt_mask = torch.ones_like(infill_mask)
        corrupt_mask = infill_seed == tokenizer.mask_token_id #TODO: check that we are only corrupting non-initial hints 

    with torch.no_grad():
        samples = model.sample(
            infill_seed = infill_seed, 
            infill_mask = infill_mask, 
            corrupt_mask = corrupt_mask, 
            num_samples = num_samples,
            guidance_kwargs = copy.deepcopy(guidance_kwargs)
        )
        samples = [tokenizer.decode(s) for s in samples]
        # evaluate and log how good the samples are 
        evaluate_samples(exp_dir, samples)



    