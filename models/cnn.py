import torch
import torch.nn as nn
from typing import Callable

class CNN(nn.Module):
    '''
    Plain 2D-CNN taking in input images shaped like `[CHANNELS, WIDTH, HEIGHT]` and mapping them to some classes.

    Parameters
    ----------
    num_classes: int
        Number of classes, thus output logits emitted
    loss_reduction: Callable
        Loss reduction, see `MeanReduction` or `SumReduction` or `HardNegativeMining`
    '''
    
    def __init__(self, num_classes: int, loss_reduction: Callable):
        super().__init__()
        
        self.num_classes = num_classes
        self.architecture = nn.Sequential(
            # first convolutional block of 32 channels, max pooling and relu
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (5, 5), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU(),
            # second convolutional block of 64 channels, max pooling and relu
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 5), padding = 'same'),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.ReLU(),
            # transform the two dimensional activation maps into one flatten array
            nn.Flatten(),
            # first fully connected layer of 2048 outputs
            nn.Linear(in_features = 7 * 7 * 64, out_features = 2048),
            nn.ReLU(),
            # second and last fully connected layer
            nn.Linear(in_features = 2048, out_features = num_classes)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index = 255, reduction = 'none')
        self.reduction = loss_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Emits `num_classes` linear logits.

        Returns
        -------
        torch.Tensor
            Logits
        '''

        return self.architecture(x)

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

        logits = self(x)
        loss = self.criterion(logits, y)
        loss = self.reduction(loss, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return logits, loss.item()

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
        
        logits = self(x)
        loss = self.criterion(logits, y)
        loss = self.reduction(loss, y)
        predicted = logits.argmax(1)
        return logits, predicted, loss.item(), (predicted == y).sum() / y.size(0)