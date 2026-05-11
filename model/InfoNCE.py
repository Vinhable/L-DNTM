import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class InfoNCE(nn.Module):
    def __init__(self, 
                 num_topics, 
                 projection_dim=768, 
                 dropout=0.1, 
                 weight_loss_InfoNCE=10.0
                ):
        super().__init__()
        self.num_topics = num_topics
        self.weight_loss_InfoNCE = weight_loss_InfoNCE
        
        self.prj_rep = nn.Sequential(nn.Linear(self.num_topics, projection_dim),
                                     nn.Dropout(dropout))
        self.prj_bert = nn.Sequential()

    def _csim(self, rep, contextual_emb):
        p_rep = self.prj_rep(rep)
        p_bert = self.prj_bert(contextual_emb)
        csim_matrix = (p_rep @ p_bert.T) / (p_rep.norm(dim=-1, keepdim=True) @ p_bert.norm(dim=-1, keepdim=True).T)
        csim_matrix = torch.exp(csim_matrix)
        csim_matrix = csim_matrix / csim_matrix.sum(dim=1, keepdim=True)
        return -csim_matrix.log()

    def _compute_loss_infonce(self, rep, contextual_emb):
        if self.weight_loss_InfoNCE <= 1e-6:
            return torch.tensor(0.0, device=rep.device)
        sim_matrix = self._csim(rep, contextual_emb)
        loss = sim_matrix.diag().mean()
        return loss * self.weight_loss_InfoNCE

    def forward(self, theta, contextual_emb):
        """
        Args:
            theta (Tensor): Topic representation from the VAE model.
            contextual_emb (Tensor): Contextual embedding from an external model.
        """
        loss_infonce = self._compute_loss_infonce(theta, contextual_emb)
        
        return {
            'loss_infonce': loss_infonce
        }