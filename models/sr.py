import torch
import torch.nn as nn
from typing import Tuple

class SimpleRepresentationEncoder(nn.Module):

    def __init__(self, num_latent_features: int):
        super().__init__()
        
        self.num_latent_features = num_latent_features
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU(),
        )
        
        self.mean = nn.Linear(in_features = 7 * 7 * 64, out_features = num_latent_features)
        self.stddev = nn.Sequential(
            nn.Linear(in_features = 7 * 7 * 64, out_features = num_latent_features),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        
        mean = self.mean(x)
        stddev = self.stddev(x)

        if self.training:
            eps = torch.randn_like(mean)
            z = mean + eps * stddev
        else:
            z = mean
        
        return z, mean, stddev

class SimpleRepresentationClassifier(torch.nn.Module):
    def __init__(
        self, 
        num_latent_features: int, 
        num_classes: int,
        beta_l2: float = 1e-2,
        beta_kl: float = 1e-3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.encoder = SimpleRepresentationEncoder(num_latent_features)
        
        self.classifier = nn.Linear(in_features = num_latent_features, out_features = num_classes)
        
        self.label_marginal_mean = nn.Parameter(torch.zeros((num_classes, num_latent_features)))
        self.label_marginal_stddev = nn.Parameter(torch.ones((num_classes, num_latent_features)))
        
        self.beta_l2 = beta_l2
        self.beta_kl = beta_kl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mean, stddev = self.encoder(x)

        logits = self.classifier(z)

        return logits, z, mean, stddev

    def step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, float]:
        logits, z, mean, stddev = self(x)
        loss = self.criterion(logits, y, z, mean, stddev)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return logits, loss.item()
    
    def criterion(self, logits, y, z, mean, stddev) -> torch.Tensor:
        variance = stddev.square()
        r_mean = self.label_marginal_mean[y]
        r_variance = self.label_marginal_stddev[y].square()

        ce_loss = nn.functional.cross_entropy(logits, y, reduction = 'mean')

        l2_regularization = z.square().sum(dim = 1).mean()

        kl_regularization = (
            (
                (r_mean - mean).square() / r_variance + 
                variance / r_variance +
                r_variance.log() - variance.log()
            ).sum(dim = 1) - self.encoder.num_latent_features
        ).mean() / 2

        loss = ce_loss + self.beta_l2 * l2_regularization + self.beta_kl * kl_regularization

        # if self.training: print(f'{ce_loss:.5f}, {l2_regularization:.5f}, {kl_regularization:.5f} -> {loss:.5f}')

        return loss
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        logits, z, mean, stddev = self(x)
        loss = self.criterion(logits, y, z, mean, stddev)
        predicted = logits.argmax(1)
        return logits, predicted, loss.item(), (predicted == y).sum() / y.size(0)