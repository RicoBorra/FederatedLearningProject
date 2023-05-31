import torch
import torch.nn as nn
from typing import Tuple

class SimpleRepresentationEncoder(nn.Module):
    '''
    Feature extractor which transforms a sample image into
    a simple representation. Basically it works as a variational
    autoencoder.

    Notes
    -----
    Since this acts as a variational autoencoder, it learns a
    conditional distribution p(z|x) := N(z; mean(x), diag(stddev(x)^2)) 
    where x is sample image and z the encoding. 
    Besides encoding z is sampled from N(z; mean(x), diag(stddev(x)^2))
    whose mean and standard deviation is learned by a 
    convolutional neural network.
    '''

    def __init__(self, num_latent_features: int):
        '''
        Constructs a simple representation encoder.

        Parameters
        ----------
        num_latent_features: int
            Size of encoded sample in latent space
        '''
        
        super().__init__()
        # dimension of encoding
        self.num_latent_features = num_latent_features
        # first convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU()
        )
        # second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU(),
        )
        # this layer extracts the mean vector of the normal distribution p(z|x)
        self.mean = nn.Linear(in_features = 7 * 7 * 64, out_features = num_latent_features)
        # this layer extracts the sigma diagonal vector of the normal distribution p(z|x)
        # since the covariance matrix is diagonal we simply extract its diagonal
        # softplus somoothly maps to a positive standard deviation
        self.stddev = nn.Sequential(
            nn.Linear(in_features = 7 * 7 * 64, out_features = num_latent_features),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Encodes input in simple representation format.

        Parameters
        ----------
        x: torch.Tensor
            Sample image

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Latent space encoding, mean and standard deviation of p(z|x)
        '''

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # extracts mean = mu, stddev = sigma of distribution p(z|x) ~ N(mu, diag(sigma^2))
        mean = self.mean(x)
        stddev = self.stddev(x)
        # on training performs parametrization trick
        if self.training:
            eps = torch.randn_like(mean)
            z = mean + eps * stddev
        # when making inference returns the mean of p(z|x) to guarantee a deterministic output
        else:
            z = mean
        # yields encoding and distribution parameters
        return z, mean, stddev

class SimpleRepresentationClassifier(torch.nn.Module):
    '''
    Compound classifier which learns two conditional distributions given each input (x, y) and
    is trained on a simple representation z of x and the corresponding label y.

    Notes
    -----
    First, the encoder distribution is p(z|x) which is a normal N(z; mean(x), diag(stddev(x)^2)).
    Secondly, a conditional r(z|y) is learnt as a categorical normal N(z; mean(y), diag(stddev(y)^2))
    on the basis of the target label. Given some x, the distribution are optimized in order to
    minimize the Kullback-Leibler divergence loss KL(p(z|x) || r(z|y)) and the L2 loss ||z||^2 which
    are the two regularization terms providing stability to the loss landscape of the classifier.
    '''

    def __init__(
        self, 
        num_latent_features: int, 
        num_classes: int,
        beta_l2: float = 1e-3,
        beta_kl: float = 1e-3
    ):
        '''
        Constructs a simple representation classifier.

        Parameters
        ----------
        num_latent_features: int
            Latent space encoding shape
        num_classes: int
            Number of target classes
        beta_l2: float = 1e-3
            Lagrangian weight of L2 regularization loss of z sampled from p(z|x)
        beta_kl: float = 1e-3
            Lagrangian weight of KL regularization loss of p(z|x) and r(z|y)
        '''
        
        super().__init__()
        self.num_classes = num_classes
        # variational autoencoder extracting a simple representation
        self.encoder = SimpleRepresentationEncoder(num_latent_features)
        # the classifier is a plain vanilla linear layer
        self.classifier = nn.Linear(in_features = num_latent_features, out_features = num_classes)
        # parameters of distribution r(z|y)
        self.label_marginal_mean = nn.Parameter(torch.zeros((num_classes, num_latent_features)))
        self.label_marginal_stddev = nn.Parameter(torch.ones((num_classes, num_latent_features)))
        # regularization weights
        self.beta_l2 = beta_l2
        self.beta_kl = beta_kl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Computes logits of simple representation z of x through the classifier.

        Parameters
        ----------
        x: torch.Tensor
            Sample image

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Logits, latent space encoding, mean and standard deviation of p(z|x)
        '''

        # encodes x as z and retrives parameters of conditional p(z|x)
        z, mean, stddev = self.encoder(x)
        # compute logits of classifier
        logits = self.classifier(z)
        # yields logits, encoding and parameters
        return logits, z, mean, stddev

    def step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> tuple[torch.Tensor, float]:
        '''
        Trains the model on a single batch and updates parameters.

        Parameters
        ----------
        x: torch.Tensor
            Input images
        y: torch.Tensor
            Input target labels 
        optimizer: torch.optim.Optimizer
            Optimizer

        Returns
        -------
        tuple[torch.Tensor, float]
            Linear logits and reduced loss
        '''

        logits, z, mean, stddev = self(x)
        # loss given by cross entropy and L2 and KL regularization terms
        loss = self.criterion(logits, y, z, mean, stddev)
        # gradient computation and weights update
        optimizer.zero_grad()
        loss.backward()
        # clip gradient norm to avoid errors
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
        optimizer.step()
        # returns outputs and scalar loss
        return logits, loss.item()
    
    def criterion(self, logits, y, z, mean, stddev) -> torch.Tensor:
        # variance of p(z|x)
        variance = stddev.square()
        # mean and variance of r(z|y)
        r_mean = self.label_marginal_mean[y]
        r_variance = self.label_marginal_stddev[y].square()
        # cross entropy loss of classifier trained on (z, y)
        ce_loss = nn.functional.cross_entropy(logits, y, reduction = 'mean')
        # regularization term, namely squared frobenius norm of z
        l2_regularization = z.square().sum(dim = 1).mean()
        # closed form of kl divergence between p(z|x) and r(z|y)
        # this closed form is simply the version of KL adapted
        # to two multivariate normals
        kl_regularization = (
            (
                (r_mean - mean).square() / r_variance + 
                variance / r_variance +
                r_variance.log() - variance.log()
            ).sum(dim = 1) - self.encoder.num_latent_features
        ).mean() / 2
        # loss is the sum of three terms
        loss = ce_loss + self.beta_l2 * l2_regularization + self.beta_kl * kl_regularization
        # result
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

        logits, z, mean, stddev = self(x)
        loss = self.criterion(logits, y, z, mean, stddev)
        predicted = logits.argmax(1)
        return logits, predicted, loss.item(), (predicted == y).sum() / y.size(0)