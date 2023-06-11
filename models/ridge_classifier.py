import torch
import torch.nn as nn
import torch.nn.functional as F 
from typing import Callable

class RidgeClassifier(nn.Module):
    '''
    Simple logistic regression model taking 2D images in input shaped like `[CHANNELS, WIDTH, HEIGHT]`
    and emitting `num_classes` logits. 
    '''

    def __init__(self, num_inputs: int, num_classes: int):
        '''
        Initializes a logistic regression classifier.

        Parameters
        ----------
        num_inputs: int
            Number of input features (in total, so `CHANNELS * WIDTH * HEIGHT`)
        num_classes: int
            Number of classes
        loss_reduction: Callable
            Loss reduction, see `MeanReduction` or `SumReduction` or `HardNegativeMining`
        '''

        super().__init__()

        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features = num_inputs, out_features = num_classes, bias = False)
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Emits `num_classes` linear logits.

        Returns
        -------
        torch.Tensor
            Logits
        '''

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

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
        y_binarized = (F.one_hot(y, num_classes=62)*2)-1
        y_binarized = y_binarized.type(torch.float32)
        logits = self(x)
        loss = self.criterion(logits, y_binarized)
        predicted = logits.argmax(1)
        return logits, predicted, loss.item(), (predicted == y).sum() / y.size(0)
