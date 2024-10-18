import sys 
import os 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import tqdm 
import hydra 
import numpy as np 
from scipy.stats import spearmanr 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import pytorch_lightning as pl 

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertEmbeddings,
    BertOnlyMLMHead
)
from trainer import BaseModel
from seq_models.regression import (RegressionHead)
from seq_models.net_utils import timestep_embedding
from guided_discrete.value.model import GPTConfig, GPT
from supervised.utils import restore_checkpoint
import logging 
class MLMDiffusionTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 dropout=0,
                 bert_config_name='bert-base-uncased',
                 target_channels=2,
                 discr_stop_grad=True,
                 num_hidden_layers=None,
                 num_attention_heads=None):
        super().__init__()

        config = AutoConfig.from_pretrained(bert_config_name)
        config.hidden_dropout_prob = dropout 
        config.vocab_size = vocab_size 
     
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers
        if num_attention_heads is not None:
            config.num_attention_heads = num_attention_heads 

        self.target_channels = target_channels
        self.vocab_size = vocab_size
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertOnlyMLMHead(config)

        self.time_embed_dim = config.hidden_size 
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, 4* config.hidden_size),
            nn.SiLU(),
            nn.Linear(4*config.hidden_size, config.hidden_size)
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if target_channels > 0:
            self.regression_head = RegressionHead(
                config,
                target_channels,
                stop_grad = discr_stop_grad
            )        
    def forward(
        self,
        corruputed_ids,
        timesteps,
        attn_mask=None,
        token_embed=None
    ):
        if token_embed is None:
            token_embed = self.embeddings(input_ids=corruputed_ids)
        
        time_embed = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))
        time_embed = time_embed.unsqueeze(1).expand(-1, token_embed.size(1), -1)
        embed = self.dropout(self.LayerNorm(token_embed + time_embed))

        # sequence_output is the output hiddens from the BERT encoder
        sequence_output = self.encoder(embed, encoder_attention_mask=attn_mask)[0]
        # prediction_scores is the logits over the vocab tokens 
        prediction_scores = self.cls(sequence_output)

        out = {
            "logits": prediction_scores,
            "sequence_output": sequence_output,
            "embeds": token_embed
        }
        return out 

class MLMDiffusion(BaseModel):
    def __init__(
        self,
        network,
        noise_schedule,
        optimizer,
        lr_scheduler
    ):
        super().__init__()
        self.network = hydra.utils.instantiate(network)
        self.noise_schedule = hydra.utils.instantiate(noise_schedule)
        self.opt = hydra.utils.instantiate(optimizer, params=self.parameters())
        self.lr_scheduler = None 
        if lr_scheduler:
            self.lr_scheduler = hydra.utils.instantiate(lr_scheduler, self.opt)

    def init_value_model(self):
        device = 'cuda:0' #TODO: figure out which device to use 
        #TODO: make this a parameter and compatible with hydra configs
        checkpoint_path= '/home/justin/Desktop/Code/sudoku-rl/guided_discrete/value/results/09/22/2024:22:45:42/best.pth'

        # load trained value function 
        model_conf = GPTConfig(vocab_size=9, block_size=81, n_head=4, n_embd=128, num_classes=9,\
                            n_recur=8, n_layer=1)
        model = GPT(model_conf)
        loaded_state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(loaded_state['model'], strict=False)
        model.to(device)
        model.eval()
        self.value_model = model 

    def freeze_for_discriminative(self):
        '''
        Freeze the network parameters and the value function parameters 
        '''
        for _, p in enumerate(self.network.parameters()):
            p.requires_grad_(False)
        
        for _, p in enumerate(self.value_model.parameters()):
            p.requires_grad_(False)
        
    def forward(
        self,
        input_ids,
        corrupt_mask,
        attn_mask,
        labels=None,
        return_by_timestep=False 
    ):
        #TODO: remove debugging inference 
        if not hasattr(self, '_forward_counter'):
            self._forward_counter = 0
        if self._forward_counter == 100 or self._forward_counter == 0:
            pass 
            #breakpoint()
        self._forward_counter += 1

        timesteps = self.noise_schedule.timesteps 
        t = torch.randint(
            timesteps, 
            size=(input_ids.shape[0],),
            device=input_ids.device,
            dtype=torch.int64
        )
        # with probability Beta_t convert input tokens to mask token 
        corrupt_ids, corrupt_mask = (
            self.noise_schedule.corrupt(input_ids, t, corrupt_mask)
        )
        
        model_output = self.network(
            corrupt_ids,
            t,
            attn_mask 
        )
        logits = model_output['logits']
        hiddens = model_output['sequence_output']

        loss_fct = nn.CrossEntropyLoss(reduction='none') 
        nll = loss_fct(logits.view(-1, logits.shape[-1]), input_ids.view(-1).long())
        nll = nll.view(*input_ids.shape[:2])

        loss_mask = attn_mask * corrupt_mask #for our case, equivalent to corrupt mask 

        denom = loss_mask.sum(dim=-1)
        denom[denom == 0] = 1
        
        # only consider the loss on tokens that were corrupted to [MASK] 
        nll = (nll * loss_mask).sum(dim=-1) / denom
        accuracy = ((logits.argmax(-1) == input_ids) * loss_mask).sum(dim=-1) / denom
        loss = nll.mean()

        out = {}
        out['loss'] = loss.mean()
        out['accuracy'] = accuracy.mean()
        out['nll'] = nll.mean()

        if labels is not None:
            raise ValueError() #this shouldn't happen for now 
            pred_labels = self.network.regression_head(hiddens.detach())
            regression_loss = (pred_labels - labels).pow(2)
            out["regression_mse"] = regression_loss.mean()
            out["regression_spearman"] = spearmanr(
                pred_labels[:,0].detach().cpu().numpy(),  
                labels[:,0].detach().cpu().numpy(),
            ).correlation
        
        if not return_by_timestep:
            return out 
        
        num_buckets = 4
        step_size = timesteps // num_buckets
        for t_lower in np.arange(0, timesteps, step_size):
            t_upper = t_lower + step_size 
            t_mask = (t > t_lower) * (t < t_upper)

            tag = f"accuracy_{t_lower}-{t_upper}"
            out[tag] = accuracy[t_mask].mean()
            '''
            if labels is not None:
                tag = f"regression_mse_{t_lower}-{t_upper}"
                out[tag] = regression_loss[t_mask].mean()
            '''
        return out 
    def infill_hints(
            self,
            one_hot_board: torch.Tensor, # one hot over digits 0-8 (batch_size, 81, 9)
            orig_sequence: torch.Tensor, # sequence of ids (1, 81),
            infill_mask, # sequence of booleans, True means can replace i.e non-given hint False means can't replace (1,81)
            tokenizer
    ):
        '''
        Take the original sequence and infill all of the initial hints (infill_mask=False) in the one hot board and return
            modified one hot board 

        NOTE: WARNING This assumes that the [MASK] token has id 0  
        '''
        assert tokenizer.convert_tokens_to_ids('[MASK]') == 0
        infill_mask = infill_mask.to(one_hot_board.device)
        

        orig_sequence = torch.flatten(orig_sequence).to(one_hot_board.device)
        orig_digits = orig_sequence - 1 # convert ids to digits 0-8
        # hack: everywhere that orig_sequence had a mask, convert it to value 1 (this doesn't matter because we won't use orig_digits where it had mask)
        orig_digits = torch.where(infill_mask, torch.ones_like(orig_digits), orig_digits)
        orig_digits = orig_digits.expand(size=(one_hot_board.shape[:2]))

        infill_mask = torch.flatten(infill_mask)
        infill_mask = infill_mask[None, :, None]
        infill_mask = infill_mask.expand_as(one_hot_board) 

        infilled_one_hot_board = torch.where(infill_mask, one_hot_board, F.one_hot(orig_digits, num_classes=9))
        return infilled_one_hot_board

    def guidance_steps(
        self,
        model_output,
        t,
        attn_mask,
        infill_mask,
        orig_ids, #sequence of token ids
        tokenizer,
        return_max_hidden=False, # boolean determines whether to return the best hidden found in search  
        guidance_layer="last",
        step_size=1,
        stability_coef=0.01, #1e-2
        num_steps=25,
    ):
        print(f"KL coef {stability_coef}")
        logger = logging.getLogger(__name__)
        logger.info("This is a log message from the model")

        kl_loss = torch.nn.KLDivLoss(log_target=True)

        logits = model_output['logits']
        if guidance_layer == "last":
            hidden = model_output['sequence_output']
        elif guidance_layer == 'first':
            raise ValueError("Incorrect implementation for first")
            hidden = model_output['embeds']
        else:
            raise NotImplementedError()
        delta = torch.nn.Parameter(torch.zeros_like(hidden), requires_grad=True)
        #optimizer = torch.optim.Adagrad([delta], lr=step_size)
        optimizer = torch.optim.Adam([delta], lr=1e-2)
        #optimizer = torch.optim.AdamW([delta], lr=5e-3, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100, verbose=True)

        max_value = 0
        max_hidden = hidden 

        with torch.enable_grad():
            for iter in range(num_steps):
                #NOTE: value function depends on the unpertubed hint logits.... 
                # needs to be some infill function 
                grad_infill_mask = (infill_mask.detach() * 1.0).requires_grad_(True)
                h_current = hidden + grad_infill_mask.unsqueeze(-1)*delta 

                if guidance_layer == "last":
                    # calculate the value prediction. NOTE: Vocab must have [MASK] at id 0 
                    new_logits = self.network.cls(h_current) #(batch_size, 81, vocab_size)
                    new_logits_digits = new_logits[:, :, 1:]
                    #sampled_noisy_one_hot = F.gumbel_softmax(new_logits_digits, hard=True)
                    sampled_noisy_one_hot = F.gumbel_softmax(new_logits_digits, tau=0.1, hard=False)
                    # May need to replace with the right hint encodings 
                    #NOTE: With like 95+% probability, infilled_one_hot.argmax(-1) is same as sampled_noisy_one_hot.argmax(-1), so we can ignore infilling step
                    value_device = self.value_model.pos_emb.device
                    sampled_noisy_one_hot = sampled_noisy_one_hot.to(value_device)

                    #infilled_one_hot = self.infill_hints(sampled_noisy_one_hot, orig_ids, infill_mask, tokenizer)
                    #print(f"They are equal: {torch.equal(infilled_one_hot.argmax(-1), sampled_noisy_one_hot.argmax(-1))}")

                    value_hat = self.value_model(sampled_noisy_one_hot.to(value_device))
                    if value_hat > max_value:
                        max_value = value_hat 
                        max_hidden = h_current  
                    
                elif guidance_layer == 'first':
                    raise ValueError("Not correctly implemented for case first")
                
                new_log_softmax = F.log_softmax(new_logits, dim=-1)
                log_softmax = F.log_softmax(logits, dim=-1)
                kl = kl_loss(new_log_softmax, log_softmax)

                #if iter % 10 == 0:
                    #print(f"KL: {kl} Value: {value_hat} Delta mean squared {torch.mean(delta.data ** 2)} Max Value {max_value} \n")
                loss = -value_hat + kl*stability_coef  
                #loss = -value_hat 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step(loss)
        #breakpoint()
        if return_max_hidden:
            logits = self.network.cls(max_hidden)
        else:
            logits = self.network.cls(hidden + delta.data) #NOTE: this is incorrect for case "first"
        return logits 
    
    def sample(
        self,
        infill_seed, 
        infill_mask,
        corrupt_mask,
        num_solutions_generate,
        tokenizer,
        guidance_kwargs=None,
        return_best=True 
    ):
        '''
        Given one starting board, generates num_solutions_generate many solutions and returns one solution (the best) 

        infill_seed: sequence to complete, contains [MASK] token ids. Shape (81, ) 
        infill_mask: vector of booleans (of shape infill_seed) where True means that token can be replaced, False means keep the original token. Shape (81, ) 
        corrupt_mask: vector of booleans where True means that the token can be corrupted, False means not. Shape (81, ) 

        For reference please refer to 2305.20009 Appendix section B Algo 1&2
        '''
        assert tokenizer.convert_tokens_to_ids('[MASK]') == 0 #requires mask id to be 0
        self.init_value_model() 
        self.freeze_for_discriminative() # freeze network and value function parameters

        assert len(infill_seed.shape) == 1, "only take on starting board of shape (81,)"
        device = next(self.parameters()).device 
        infill_mask = infill_mask[None, :]
        corrupt_mask = corrupt_mask[None, :]
        gt_vals = infill_seed[None]

        indices = list(range(self.noise_schedule.timesteps))[::-1]

        # corrupt the sequence based on the final timestep t only where corrupt_mask is True 
        t = torch.tensor([indices[0]], device=device)
        noisy_gt = self.noise_schedule.corrupt(gt_vals, t)[0]
        noisy_gt = torch.where(corrupt_mask, noisy_gt, gt_vals)

        shape = (num_solutions_generate, infill_seed.shape[0])
        
        # x starts as tensor filled with all mask tokens where infill mask is True (rest are initial hints) 
        x = self.noise_schedule.sample_prior(shape, device)
        x = torch.where(infill_mask, x, noisy_gt)
        #TODO: pdb and check that the initial hints are preserved 
        attn_mask = torch.ones_like(infill_mask, dtype=torch.bool)
        
        if guidance_kwargs is not None:
            return_best = guidance_kwargs.pop("return_best", False)
        
        return_best_logits = guidance_kwargs.pop("return_best_logits", False) if guidance_kwargs is not None else False 
        
        traj = []
        
        # iterate over diffusion timesteps 
        for i in tqdm.tqdm(indices):
            t = torch.tensor([i] * shape[0], device=device)

            with torch.no_grad():
                model_output = self.network(x,t, attn_mask)
            
            logits = model_output['logits']

            if guidance_kwargs is not None:
                guided_logits = self.guidance_steps(
                    model_output, t, attn_mask, infill_mask, gt_vals, tokenizer,return_best_logits,
                    **guidance_kwargs
                )
                diff_with_guidance = torch.mean((logits - guided_logits)**2)
                print(f"Difference with guidance {diff_with_guidance}")
                logits = guided_logits
            
            # generate a denoised sample based on my noisy sequence x 
            x = Categorical(logits=logits).sample()
            clean_x = x.clone()

            if i != indices[-1]:
                # renoise x according to my "next" timestep t only where we don't have initial hints; 
                # ensure that x has original hints tokens preserved with the infill_mask 
                x = self.noise_schedule.corrupt(x,t,infill_mask)[0]
                # noisy_gt is effectively useless but important thing is that all the initial hints are preserved 
                noise_t = torch.tensor([i-1]*shape[0], device=device)
                noisy_gt = self.noise_schedule.corrupt(gt_vals, noise_t[:1])[0]
                noisy_gt = torch.where(corrupt_mask.bool(), noisy_gt, gt_vals)
                x = torch.where(infill_mask, x, noisy_gt)
  
            # replace the initial hints into the generated denoised sample 
            pred_ids = torch.where(infill_mask.squeeze(-1), clean_x, infill_seed[None])
            pred_ids = pred_ids.cpu()
            
            traj.append(pred_ids)
        
        if return_best: #return the best of the final diffusion output
            samples = traj[-1].to(self.value_model.pos_emb.device)
            pred_boards = samples - 1
            one_hot_pred_boards = F.one_hot(pred_boards, num_classes=9).float()
            value_scores = self.value_model(one_hot_pred_boards).squeeze()
            best_sample_index = torch.argmax(value_scores)
            return samples[best_sample_index].cpu().numpy()
        else:
            return traj[-1][0]




        













