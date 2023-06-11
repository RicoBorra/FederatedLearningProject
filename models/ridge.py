import torch
import torch.nn as nn
from typing import Callable

class RidgeRegression(nn.Module):
    '''
    Simple ridge regression model emitting `num_classes` logits. 
    '''

    def __init__(self, num_inputs: int, num_classes: int, device: torch.device):
        '''
        Initializes a logistic regression classifier.

        Parameters
        ----------
        num_inputs: int
            Number of input features (in total, so `CHANNELS * WIDTH * HEIGHT`)
        num_classes: int
            Number of classes
        '''

        super().__init__()

        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.beta = nn.Parameter(torch.empty(size = (num_inputs + 1, num_classes), device = device, dtype = torch.float32))
        self.criterion = nn.MSELoss(reduction = 'mean')
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Emits `num_classes` linear logits.

        Returns
        -------
        torch.Tensor
            Logits
        '''

        return x @ self.beta

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

        return None, None

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

        # appends one column (for bias) to features matrix
        ones = torch.ones((x.shape[0], 1), device = self.device)
        x = torch.cat((ones, x), dim = -1)
        # center class label in range [-1, 1]
        y_binarized = (torch.nn.functional.one_hot(y, num_classes = self.num_classes) * 2) - 1
        y_binarized = y_binarized.type(torch.float32)
        # predicts as usual
        logits = self(x)
        loss = self.criterion(logits, y_binarized)
        predicted = logits.argmax(1)
        return logits, predicted, loss.item(), (predicted == y).sum() / y.size(0)
