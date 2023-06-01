from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from typing import Any, Tuple

from .evaluation import FederatedMetrics

class Client(object):
    '''
    This class simulates a client with its own dataset who represents a terminal user, 
    furthermore it can be either a training client, validation client or testing client.
    '''

    def __init__(
        self,
        args: Any,
        name: str,
        dataset: Subset,
        device: torch.device,
        validator: bool = False,
    ):
        '''
        Initializes a user client.

        Parameters
        ----------
        args: Any
            Command line arguments of the simulation
        name: str
            Identifier name of the client
        dataset: Subset
            Local dataset used by the client for training or validation or testing
        device: torch.device
            Device, cuda gpu or cpu, on which computation is performed
        validator: bool
            Tells if it is a validator (False by default)
        '''
        
        self.args = args
        self.name = name
        self.dataset = dataset
        self.validator = validator
        self.device = device
        self.loader = DataLoader(dataset, batch_size = args.batch_size)

    def train(self, algorithm: Any, model: nn.Module) -> Any:
        '''
        Runs federated `algorithm` for training to optimize central server `model`
        on local dataset.

        Parameters
        ----------
        algorithm: Any
            Algorithm to be run on client at each round, see `FedAlgorithm` descendants
        model: nn.Module
            Model initialized with central server parameters
            at the beginning of the round

        Returns
        -------
        Any
            State update collected by central server algorithm
        '''

        # fails if training is invoked on a validator or tester
        assert not self.validator
        # local training handled by appropriate algorithm and
        # then updated weights are returned
        return algorithm.visit(client = self, model = model)
    
    def validate(self, model: nn.Module, metrics: FederatedMetrics):
        '''
        Evaluates central model performance on local client dataset and updates metrics.

        Notes
        -----
        The local dataset, limited in size, is fed in one pass to the
        model to speed up parallelization.

        Parameters
        ----------
        model: nn.Module
            Model initialized with central server parameters
            at the beginning of the round
        metrics: FederatedMetrics
            Metrics of the central server which are updated
            with evaluation results
        '''

        # note that torch.no_grad is invoked a priori by the server
        # so no need here
        loss, logits, y = self.evaluate(model)
        # performance metrics are updated with outputs and targets
        metrics.update(logits, y, loss)

    def evaluate(self, model: nn.Module) -> Tuple[float, torch.Tensor, torch.Tensor]:
        '''
        Evaluates central model performance on local client dataset.

        Notes
        -----
        The local dataset, limited in size, is fed in one pass to the
        model to speed up parallelization.

        Parameters
        ----------
        model: nn.Module
            Model initialized with central server parameters
            at the beginning of the round

        Returns
        -------
        Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, float]
            Tuple with loss, logits, y, predictions and accuracy
        '''

        loader = DataLoader(self.dataset, batch_size = len(self.dataset))
        x, y = next(iter(loader))
        x = x.to(self.device)
        y = y.to(self.device)
        # validation mode
        model.eval()
        # note that torch.no_grad is invoked a priori by the server so no need here
        logits, _, loss, _ = model.evaluate(x, y)

        ## FIXME checks
        # assert torch.isfinite(logits).all() and np.isfinite(loss)

        # yields outputs
        return loss, logits, y

def construct(user_datasets: dict[str, list[tuple[str, Subset]]], device: torch.device, args: Any) -> dict[str, list[Client]]:
    '''
    Constructs groups of clients from the groups of datasets.

    Parameters
    ----------
    user_datasets: dict[str, list[tuple[str, Subset]]]
        Dictionary of clients' datasets, with entries such as `training`,
        `validation` or `testing` 
    device: torch.device
        Device used by clients for the simulation
    args: Any
        Command line arguments of the simulation

    Returns
    -------
    dict[str, list[Client]]
        Dictionary with groups of clients, like `training`,
        `validation` or `testing` clients
    '''
    
    return {
        group: [
            # client is set as validator when he is not assigned to 'training' group
            Client(name = name, dataset = data, validator = (group != 'training'), device = device, args = args) 
            for name, data in datasets 
        ]
        # group can either be 'training','validation' or 'testing'
        for group, datasets in user_datasets.items()
    }