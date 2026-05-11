import torch
import torch.nn as nn
import numpy as np

class CKALoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, SH, TH): 
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.view(-1, dS).to(SH.device, torch.float64)
        TH = TH.view(-1, dT).to(SH.device, torch.float64)
        
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
                
        num = torch.norm(SH.t().matmul(TH), 'fro')
        den1 = torch.norm(SH.t().matmul(SH), 'fro') + self.eps
        den2 = torch.norm(TH.t().matmul(TH), 'fro') + self.eps
        
        return 1 - num/torch.sqrt(den1*den2)

class CKA(nn.Module):
    def __init__(self, weight_cka=300.0):
        super().__init__()
        self.weight_cka = weight_cka
        self.cka_loss_fn = CKALoss(eps=1e-8)

    def forward(self, theta, doc_embedding):
        """
        Args:
            theta (Tensor): Topic representation from the VAE model.
            doc_embedding (Tensor): Document embedding from the external model.
        """
        cka_loss = self.cka_loss_fn(theta, doc_embedding)
        cka_loss_weighted = cka_loss * self.weight_cka
        
        return {
            'loss_cka': cka_loss_weighted
        }