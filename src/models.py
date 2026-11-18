"""Two papers, combined deliberately (not a single paper doing both):

1. Chivukula, Yang, Liu, Zhu, Zhou, "Game Theoretical Adversarial Deep
   Learning With Variational Adversaries", IEEE TKDE 2021 - the VAE as the
   adversary's generative strategy space, and the Stackelberg-game attack
   in attack.py.
2. Zhu, Hu, Zhang, Zuo, "Coupled Dictionary Learning for Unsupervised
   Feature Selection", AAAI 2016 - the CDLLayer below: a synthesis
   dictionary U that reconstructs samples (here, latent codes) and an
   analysis dictionary V that codes them, with L2,p-norm group-sparsity
   regularization on V so its row norms rank which latent dimensions
   actually matter. That ranking is then used to restrict the Stackelberg
   search to the dimensions that matter, which is also what makes the
   attack tractable instead of searching all 32 latent dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 28->14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 14->7
        )
        self.fc_mu = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 32 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7->14
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),  # 14->28
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 32, 7, 7)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def vae_loss(recon, x, mu, logvar):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


class CDLLayer(nn.Module):
    """Zhu et al. 2016: synthesis dictionary U reconstructs z as U @ (V^T z);
    analysis dictionary V analytically codes z. L2,p-norm regularization on
    V's row norms (0 < p <= 1) drives group sparsity - a row of V near zero
    means that latent dimension contributes little to a good reconstruction,
    so it's one the analysis dictionary has effectively deselected."""

    def __init__(self, latent_dim, dict_size=64, tau=0.01, p=0.7):
        super().__init__()
        self.tau = tau
        self.p = p
        self.U = nn.Parameter(torch.randn(latent_dim, dict_size) * 0.1)
        self.V = nn.Parameter(torch.randn(latent_dim, dict_size) * 0.1)
        with torch.no_grad():
            self.U /= self.U.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def forward(self, z):
        coeffs = z @ self.V          # (B, dict_size) analysis code
        z_recon = coeffs @ self.U.t()  # (B, latent_dim) synthesis reconstruction
        return z_recon, coeffs

    def cdl_loss(self, z):
        z_recon, _ = self.forward(z)
        recon_loss = F.mse_loss(z_recon, z)
        row_norms = self.V.norm(dim=1, p=2)  # (latent_dim,) - one per latent coordinate
        sparsity_loss = torch.sum(torch.pow(row_norms + 1e-8, self.p))
        return recon_loss + self.tau * sparsity_loss, recon_loss, sparsity_loss

    def feature_importance(self):
        """The analysis dictionary's row norms - the paper's own feature-
        selection criterion: a latent dimension the analysis dictionary
        needs a large row to code is one carrying real structure."""
        return self.V.detach().norm(dim=1, p=2)


class ProductCNN(nn.Module):
    """The learner: a small CNN classifier for the 10 apparel categories."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)
