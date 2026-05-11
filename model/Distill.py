import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CKA import CKA
from .InfoNCE import InfoNCE

class DistillVAE(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_topics=50,
                 en_units=200,
                 dropout=0.1,
                 w_cka=1.0,
                 w_infonce=1.0,
                 weight_cka_internal=300.0,
                 weight_infonce_internal=10.0,
                 infonce_proj_dim=768
                 ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = None

        self.beta = nn.Parameter(torch.empty(num_topics, vocab_size))
        nn.init.xavier_uniform_(self.beta)

        self.w_cka = w_cka
        self.w_infonce = w_infonce

        self.loss_fn_cka = CKA(weight_cka=weight_cka_internal)
        self.loss_fn_infonce = InfoNCE(
            num_topics=num_topics, 
            weight_loss_InfoNCE=weight_infonce_internal,
            projection_dim=infonce_proj_dim
        )

    def decode(self, theta):
        return F.softmax(torch.matmul(theta, self.beta), dim=-1)

    def forward(self, bow, doc_embedding, contextual_emb, external_theta=None):  
        if external_theta is not None:
            theta = external_theta
        else:
            theta, _, _ = self.encoder(bow)
        loss_cka_dict = self.loss_fn_cka(theta, doc_embedding)
        loss_infonce_dict = self.loss_fn_infonce(theta, contextual_emb)
        loss_cka = loss_cka_dict['loss_cka']
        loss_infonce = loss_infonce_dict['loss_infonce']
        distill_loss = (self.w_cka * loss_cka) + (self.w_infonce * loss_infonce)

        rst_dict = {
            'loss': distill_loss,
            'loss_cka': loss_cka,
            'loss_infonce': loss_infonce,
        }
        
        return rst_dict