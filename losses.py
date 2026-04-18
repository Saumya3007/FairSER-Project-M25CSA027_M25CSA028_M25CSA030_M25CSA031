# losses.py — FocalLoss + CLUES with class weights
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight          # torch.Tensor shape [num_classes]
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # label-smoothed CE first
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class CLUESLoss(nn.Module):
    """
    CE + contrastive fairness loss.
    Pulls same-emotion embeddings together across language/gender groups.
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.1,
                 alpha=0.3, temperature=0.07):
        super().__init__()
        self.focal       = FocalLoss(gamma, weight, label_smoothing)
        self.alpha       = alpha        # weight on contrastive term
        self.temperature = temperature

    def forward(self, logits, embeddings, targets):
        ce_loss = self.focal(logits, targets)

        emb = F.normalize(embeddings, dim=-1)
        sim = torch.matmul(emb, emb.T) / self.temperature  

        mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos = mask.sum(dim=1).clamp(min=1)
        contrastive = -(log_prob * mask).sum(dim=1) / n_pos
        contrastive_loss = contrastive.mean()

        return ce_loss + self.alpha * contrastive_loss, ce_loss, contrastive_loss
