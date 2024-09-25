import torch
import torch.nn as nn 
import torch.nn.functional as F 

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertPooler,
    BertEncoder,
    BertEmbeddings,
    BertOnlyMLMHead
)
from seq_models.net_utils import timestep_embedding 

class RegressionHead(nn.Module):
    def __init__(
        self,
        config,
        out_channels,
        stop_grad=True
    ):
        super().__init__()
        self.pooler = BertPooler(config)

        classifier_dropout_prob = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.regression_dropout = nn.Dropout(classifier_dropout_prob)
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

        self.stop_grad = stop_grad 
    
    def forward(self, sequence_output, attn_mask=None):
        if self.stop_grad:
            sequence_output = sequence_output.detach()

        pooled_output = sequence_output.mean(1)
        pooled_output = self.regression_dropout(pooled_output)
        regression_pred = self.regression_head(pooled_output)
        return regression_pred

