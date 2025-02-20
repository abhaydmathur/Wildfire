from torch import nn
import torch
import torch.nn.functional as F


class BetaVAELoss(nn.Module):
    def __init__(self, beta=1):
        super(BetaVAELoss, self).__init__()
        self.beta = beta

    def forward(self, x, recon_x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss


criterion_vae = BetaVAELoss(beta=1)
