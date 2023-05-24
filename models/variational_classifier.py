import torch
import torch.nn as nn
from typing import Tuple

class VariationalEncoder(nn.Module):

    def __init__(self, num_representation_outputs: int):
        super().__init__()
        # latent space dimension, or simple representation shape
        self.num_representation_outputs = num_representation_outputs
        # first convolutional block of 32 channels, max pooling and relu
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU()
        )
        # second convolutional block of 64 channels, max pooling and relu
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU(),
        )
        # first fully connected extracts mu, namely the mean of p(z|x)
        self.mean = nn.Linear(in_features = 7 * 7 * 64, out_features = num_representation_outputs)
        # second fully connected extracts sigma ** 2, namely the variance of p(z|x)
        self.log_variance = nn.Linear(in_features = 7 * 7 * 64, out_features = num_representation_outputs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        
        mean = self.mean(x)
        log_variance = self.log_variance(x)

        if self.training:
            eps = torch.randn_like(mean)
            z = mean + eps * torch.exp(log_variance / 2)
        else:
            z = mean
        
        return z, mean, log_variance
        

class VariationalClassifier(nn.Module):

    def __init__(
        self, 
        num_representation_outputs: int, 
        num_classes: int,
        beta_l2n: float = 1e-2,
        beta_cmi: float = 5e-4
    ):
        super().__init__()
        # number of outputs
        self.num_classes = num_classes
        # constructs simple representation z sampled from p(z|x)
        self.encoder = VariationalEncoder(num_representation_outputs)
        # classifier which is trained on (z, y) pairs
        self.classifier = nn.Linear(in_features = num_representation_outputs, out_features = num_classes)
        # mean and log variances of z|y distribution, manely r(z|y)
        self.label_marginal_mean = nn.Parameter(torch.zeros((num_classes, num_representation_outputs)))
        self.label_marginal_log_variance = nn.Parameter(torch.zeros((num_classes, num_representation_outputs)))
        # lagrangians of regularizers
        self.beta_l2n = beta_l2n
        self.beta_cmi = beta_cmi

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        z, mean, log_variance = self.encoder(x)

        logits = self.classifier(z)

        return logits, z, mean, log_variance

    def step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, float]:
        logits, z, mean, log_variance = self(x)

        loss = self.criterion(logits, y, z, mean, log_variance)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return logits, loss.item()
    
    def criterion(self, logits, y, z, mean, log_variance) -> torch.Tensor:
        
        r_mean = self.label_marginal_mean[y]
        r_log_variance = self.label_marginal_log_variance[y]
        k = z.size(1)

        ce_loss = nn.functional.cross_entropy(logits, y, reduction = 'mean')

        l2_regularization = z.square().sum(dim = 1).mean()

        kl_regularization = (
            (
                (r_mean - mean).square() * (-r_log_variance).exp() + 
                (log_variance - r_log_variance).exp() +
                (r_log_variance - log_variance) 
            ).sum(dim = 1) - k
        ).mean() / 2

        loss = ce_loss + self.beta_l2n * l2_regularization + self.beta_cmi * kl_regularization

        # print(f'{ce_loss:.5f}, {l2_regularization:.5f}, {kl_regularization:.5f} -> {loss:.5f}')

        return loss
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        '''
        Evaluates model on a single batch of data, no gradient or updates are computed.

        Parameters
        ----------
        x: torch.Tensor
            Input images
        y: torch.Tensor
            Input target labels

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, float]
            Linear logits, predicted labels, reduced loss and accuracy
        '''
        
        logits, z, mean, log_variance = self(x)
        loss = self.criterion(logits, y, z, mean, log_variance)
        predicted = logits.argmax(1)
        return logits, predicted, loss.item(), (predicted == y).sum() / y.size(0)

        
################## NOTE IMPLEMENTATION USING VARIANCE, NOT LOG(VARIANCE) ######################

class VariationalEncoderNotLogVar(nn.Module):

    def __init__(self, num_representation_outputs: int):
        super().__init__()
        # latent space dimension, or simple representation shape
        self.num_representation_outputs = num_representation_outputs
        # first convolutional block of 32 channels, max pooling and relu
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU()
        )
        # second convolutional block of 64 channels, max pooling and relu
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU(),
        )
        # first fully connected extracts mu, namely the mean of p(z|x)
        self.mean = nn.Linear(in_features = 7 * 7 * 64, out_features = num_representation_outputs)
        # second fully connected extracts sigma ** 2, namely the variance of p(z|x)
        self.variance = nn.Sequential(
            nn.Linear(in_features = 7 * 7 * 64, out_features = num_representation_outputs),
            # expects variance to be positive
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        
        mean = self.mean(x)
        variance = self.variance(x)

        if self.training:
            eps = torch.randn_like(mean)
            z = mean + eps * variance.sqrt()
        else:
            z = mean
        
        return z, mean, variance
        

class VariationalClassifierNotLogVar(nn.Module):

    def __init__(
        self, 
        num_representation_outputs: int, 
        num_classes: int,
        beta_l2n: float = 1e-2,
        beta_cmi: float = 5e-4
    ):
        super().__init__()
        # number of outputs
        self.num_classes = num_classes
        # constructs simple representation z sampled from p(z|x)
        self.encoder = VariationalEncoderNotLogVar(num_representation_outputs)
        # classifier which is trained on (z, y) pairs
        self.classifier = nn.Linear(in_features = num_representation_outputs, out_features = num_classes)
        # mean and variances of z|y distribution, manely r(z|y)
        self.label_marginal_mean = nn.Parameter(torch.zeros((num_classes, num_representation_outputs)))
        self.label_marginal_variance = nn.Parameter(torch.ones((num_classes, num_representation_outputs)))
        # lagrangians of regularizers
        self.beta_l2n = beta_l2n
        self.beta_cmi = beta_cmi

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        z, mean, variance = self.encoder(x)

        logits = self.classifier(z)

        return logits, z, mean, variance

    def step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, float]:
        logits, z, mean, variance = self(x)

        loss = self.criterion(logits, y, z, mean, variance)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return logits, loss.item()
    
    def criterion(self, logits, y, z, mean, variance) -> torch.Tensor:
        
        r_mean = self.label_marginal_mean[y]
        r_variance = self.label_marginal_variance[y]
        k = z.size(1)

        ce_loss = nn.functional.cross_entropy(logits, y, reduction = 'mean')

        l2_regularization = z.square().sum(dim = 1).mean()

        kl_regularization = (
            (
                (r_mean - mean).square() / r_variance + 
                variance / r_variance +
                r_variance.log() - variance.log()
            ).sum(dim = 1) - k
        ).mean() / 2

        loss = ce_loss + self.beta_l2n * l2_regularization + self.beta_cmi * kl_regularization

        # print(f'{ce_loss:.5f}, {l2_regularization:.5f}, {kl_regularization:.5f} -> {loss:.5f}')

        return loss
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        '''
        Evaluates model on a single batch of data, no gradient or updates are computed.

        Parameters
        ----------
        x: torch.Tensor
            Input images
        y: torch.Tensor
            Input target labels

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, float]
            Linear logits, predicted labels, reduced loss and accuracy
        '''
        
        logits, z, mean, variance = self(x)
        loss = self.criterion(logits, y, z, mean, variance)
        predicted = logits.argmax(1)
        return logits, predicted, loss.item(), (predicted == y).sum() / y.size(0)