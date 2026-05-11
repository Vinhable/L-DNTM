import torch
from torch import nn

class DynamicOTLoss(nn.Module):
    """
        Calculates the Optimal Transport loss (Wasserstein distance) between the topic embeddings
        of two consecutive time slices, t-1 and t.

        The loss is calculated based on the formula: loss = <C, P>, where C is the cost matrix
        and P is the optimal transport plan found using the Sinkhorn algorithm.
    """
    def __init__(self, weight_loss_ot, sinkhorn_alpha, ot_max_iter=1000, stopThr=1e-3):
        super().__init__()
        self.weight_loss_ot = weight_loss_ot
        self.sinkhorn_alpha = sinkhorn_alpha
        self.ot_max_iter = ot_max_iter
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def _pairwise_euclidean_dist(self, x, y):
        """Tính ma trận chi phí C."""
        x_sq = torch.sum(x ** 2, axis=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, axis=-1, keepdim=True).t()
        cost = x_sq + y_sq - 2 * torch.matmul(x, y.t())
        return cost.clamp(min=0) 

    def forward(self, embeds_t_minus_1, embeds_t):
        """
        Args:
            embeds_t_minus_1 (Tensor): Topic embeddings at t-1, shape [K, D].
            embeds_t (Tensor): Topic embeddings at t, shape [K, D].
        """
        if self.weight_loss_ot <= 1e-6:
            return torch.tensor(0.0, device=embeds_t_minus_1.device)

        cost_matrix = self._pairwise_euclidean_dist(embeds_t_minus_1, embeds_t)
        device = cost_matrix.device
        num_topics = cost_matrix.shape[0]

        a = (torch.ones(num_topics, 1) / num_topics).to(device)
        b = (torch.ones(num_topics, 1) / num_topics).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device)
        K = torch.exp(-cost_matrix * self.sinkhorn_alpha)
        
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.ot_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 0:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)
        loss = torch.sum(cost_matrix * transp) * self.weight_loss_ot
        
        return loss
