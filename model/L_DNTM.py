import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import Encoder
from .ETC import ETC
from .UWE import UWE
from .DynamicOT import DynamicOTLoss
from .Distill import DistillVAE
from .LLMGuider import LLMGuider

class L_DNTM(nn.Module):
    def __init__(self,
                 vocab_size, num_times, num_topics, train_time_wordfreq,
                 word_embeddings, en_units, dropout, beta_temp,
                 temperature, weight_neg, weight_pos, weight_UWE, neg_topk,
                 weight_loss_ot=1.0, ot_sinkhorn_alpha=20.0, ot_max_iter=1000, ot_warm_up=0,
                 weight_distill_loss=1.0, distill_en_units=200, distill_dropout=0.1,
                 distill_w_cka=1.0, distill_w_infonce=1.0,
                 distill_weight_cka_internal=300.0,
                 distill_weight_infonce_internal=10.0, distill_infonce_proj_dim=768,
                 llm_warm_up_epochs=150,
                 lambda_contrastive=0.0,
                 gemini_model_name="gemini-2.5-flash",
                 llm_contrastive_temperature=0.1,
                 llm_guidance_refresh_rate=10,
                 llm_top_k=15,
                 llm_history_length=3,
                 llm_max_retries=3,
                 llm_retry_delay=5,
                 idx_to_word=None,
                 word_to_idx=None,
                 llm_batch_size=5,
                 llm_log_path="./llm_logs/"                 
                ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_times = num_times
        self.num_topics = num_topics
        self.train_time_wordfreq = train_time_wordfreq
        self.ot_warm_up = ot_warm_up
        self.llm_warm_up_epochs = llm_warm_up_epochs

        encoder_args = type('Args', (object,), {
            'vocab_size': vocab_size, 'num_topic': num_topics,
            'model': type('ModelArgs', (object,), {'en1_units': en_units, 'dropout': dropout})
        })()
        self.encoder = Encoder(encoder_args)

        self.a = 1 * np.ones((1, self.num_topics)).astype(np.float32)
        mu2 = torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        var2 = torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / self.num_topics))).T + (1.0 / (self.num_topics * self.num_topics)) * np.sum(1.0 / self.a, 1)).T)
        self.register_buffer('mu2', mu2)
        self.register_buffer('var2', var2)

        self.decoder_bn = nn.BatchNorm1d(self.vocab_size, affine=False)
        self.word_embeddings = nn.Parameter(torch.from_numpy(word_embeddings).float())
        self.topic_embeddings = nn.Parameter(
            nn.init.xavier_normal_(torch.zeros(self.num_topics, self.word_embeddings.shape[1]))
            .repeat(self.num_times, 1, 1)
        )
        self.beta_temp = beta_temp
        self.ETC = ETC(self.num_times, temperature, weight_neg, weight_pos)
        self.UWE = UWE(self.ETC, self.num_times, temperature, weight_UWE, neg_topk)

        self.ot_loss = DynamicOTLoss(
            sinkhorn_alpha=ot_sinkhorn_alpha,
            ot_max_iter=ot_max_iter,
            weight_loss_ot = weight_loss_ot
        )
        self.weight_loss_ot = weight_loss_ot

        self.distill_module = DistillVAE(
            vocab_size=vocab_size, num_topics=num_topics, en_units=distill_en_units,
            dropout=distill_dropout, w_cka=distill_w_cka,
            w_infonce=distill_w_infonce, weight_cka_internal=distill_weight_cka_internal,
            weight_infonce_internal=distill_weight_infonce_internal,
            infonce_proj_dim=distill_infonce_proj_dim
        )
        self.distill_module.encoder = self.encoder 
        self.weight_distill_loss = weight_distill_loss

        if idx_to_word is None or word_to_idx is None:
            raise ValueError("idx_to_word and word_to_idx must be provided for LLMGuider.")
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

        self.llm_guider = LLMGuider(
            lambda_contrastive=lambda_contrastive,
            gemini_model_name=gemini_model_name,
            llm_contrastive_temperature=llm_contrastive_temperature,
            llm_guidance_refresh_rate=llm_guidance_refresh_rate,
            llm_top_k=llm_top_k,
            llm_history_length=llm_history_length,
            llm_max_retries=llm_max_retries,
            llm_retry_delay=llm_retry_delay,
            num_times=self.num_times,
            num_topic=self.num_topics,
            log_path=llm_log_path,
            llm_batch_size=llm_batch_size            
        )

    def get_beta(self):
        dist = self._pairwise_euclidean_dist(
            F.normalize(self.topic_embeddings, dim=-1), 
            F.normalize(self.word_embeddings, dim=-1)
        )
        beta = F.softmax(-dist / self.beta_temp, dim=1)
        return beta

    def _pairwise_euclidean_dist(self, x, y):
        x_sq = torch.sum(x ** 2, axis=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, axis=-1)
        cost = x_sq + y_sq - 2 * torch.matmul(x, y.t())
        return cost.clamp(min=0)

    def get_theta(self, x, times=None):
        theta, _, _ = self.encoder(x)
        return theta 
    
    def get_KL(self, mu, logvar):
        var = logvar.exp()
        KLD = 0.5 * ((var / self.var2 + (mu - self.mu2)**2 / self.var2 + 
                      self.var2.log() - logvar).sum(axis=1) - self.num_topics)
        return KLD.mean()

    def decode(self, theta, beta_for_docs):
        recon_logits = torch.bmm(theta.unsqueeze(1), beta_for_docs).squeeze(1)
        return F.softmax(self.decoder_bn(recon_logits), dim=-1)

    def forward(self, x, times, doc_embedding, contextual_emb, epoch=None):
        theta, mu, logvar = self.encoder(x)
        kl_theta = self.get_KL(mu, logvar)

        beta = self.get_beta()
        time_index_beta = beta.index_select(0, times.long()) if times.ndim > 0 else beta[times]
        recon_x = self.decode(theta, time_index_beta)
        NLL = -(x * recon_x.log()).sum(axis=1).mean()
        
        loss_ETC = self.ETC(self.topic_embeddings)
        loss_UWE = self.UWE(self.train_time_wordfreq, beta, self.topic_embeddings, self.word_embeddings)

        loss_cfdtm = NLL + kl_theta + loss_ETC + loss_UWE
        
        loss_ot = torch.tensor(0.0, device=x.device)
        if epoch is not None and epoch > self.ot_warm_up:
            current_loss_ot = 0.
            for t in range(1, self.num_times):
                current_loss_ot += self.ot_loss(self.topic_embeddings[t-1].detach(), self.topic_embeddings[t])
            loss_ot = current_loss_ot / (self.num_times - 1) if self.num_times > 1 else torch.tensor(0.0, device=x.device)

        distill_results = self.distill_module(
            bow=x,
            doc_embedding=doc_embedding,
            contextual_emb=contextual_emb
        )
        loss_distill = distill_results['loss']

        loss_llm = torch.tensor(0.0, device=x.device)
        if epoch is not None and epoch > self.llm_warm_up_epochs:
            self.llm_guider.update_guidance_cache(epoch, beta.detach(), self.idx_to_word)
            loss_llm = self.llm_guider.calculate_contrastive_loss(self.topic_embeddings, self.word_embeddings, self.word_to_idx)

        total_loss = (
            loss_cfdtm 
            + self.weight_loss_ot * loss_ot
            + self.weight_distill_loss * loss_distill
            + self.llm_guider.lambda_contrastive * loss_llm  
        )

        rst_dict = {
            'loss': total_loss,
            'loss_core': loss_cfdtm,
            'loss_ot': loss_ot,
            'loss_distill': loss_distill,
            'loss_llm': loss_llm, 
            'core_nll': NLL,
            'core_kl_theta': kl_theta,
            'core_etc': loss_ETC,
            'core_uwe': loss_UWE,
            'distill_cka': distill_results['loss_cka'],
            'distill_infonce': distill_results['loss_infonce']
        }
        return rst_dict
