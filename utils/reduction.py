import torch
import torch.nn as nn

class HardNegativeMining(nn.Module):
    '''
    Computes an hard negative mining aggregation on a loss tensor.
    '''

    def __init__(self, perc: float = 0.25):
        '''
        Initializes the reduction object.

        Parameters
        ----------
        perc: int
            Percentage of values on which to compute mean.
        '''

        super().__init__()

        self.perc = perc

    def forward(self, loss: torch.Tensor, _) -> torch.Tensor:
        '''
        Reduces a loss tensor to a single number.

        Parameters
        ----------
        loss: torch.Tensor
            Loss tensor
        
        Returns
        -------
        torch.Tensor
            Reduced loss
        '''

        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


class MeanReduction(object):
    '''
    Computes a mean aggregation on a loss tensor.
    '''

    def __call__(self, loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Reduces a loss tensor to a single number.

        Parameters
        ----------
        loss: torch.Tensor
            Loss tensor
        target: torch.Tensor
            Class labels
        
        Returns
        -------
        torch.Tensor
            Reduced loss
        '''

        loss = loss[target != 255]
        return loss.mean()

class SumReduction(object):
    '''
    Computes a sum aggregation on a loss tensor.
    '''

    def __call__(self, loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Reduces a loss tensor to a single number.

        Parameters
        ----------
        loss: torch.Tensor
            Loss tensor
        target: torch.Tensor
            Class labels
        
        Returns
        -------
        torch.Tensor
            Reduced loss
        '''

        loss = loss[target != 255]
        return loss.sum()
