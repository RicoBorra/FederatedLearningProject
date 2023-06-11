from abc import ABC, abstractmethod
import math
import numpy as np
import random
import torch
from typing import Any


class ClientSelection(ABC):
    '''
    Base class for client selection strategy during each round
    of the server training.
    '''
    
    def __init__(self, server: Any):
        '''
        Constructs a client selection strategy.

        Parameters
        ----------
        server: Server
            Reference to server instance
        '''
        
        # command line parameters of the simulation
        self._args = server.args
        # whole group of training clients
        self._clients = server.clients['training']
        # reference to central model of the server
        self._model = server.model
        # probability (of being selected at each round)
        # assigned to each client, not normalized
        self._importances = np.ones(len(self._clients), dtype = np.float32)

    @property
    def probabilities(self) -> np.ndarray:
        '''
        Property for retrieving client probabilities of being chosen for training.

        Returns
        -------
        np.ndarray
            Likelihoods of each client of being selected at the
            current round, not normalized
        '''
        
        return self._importances

    @abstractmethod
    def select(self) -> list[Any]:
        '''
        Selects a bunch of clients for being trained according
        to a federated policy during current round.

        Returns
        -------
        list[Client]
            List of clients selected for training
        '''
        
        raise NotImplementedError()

class UniformSelection(ClientSelection):
    '''
    This strategy selected `args.selected` random clients during
    each round with uniform probability.
    '''

    def select(self) -> list[Any]:
        '''
        Selects with uniform probability a bunch of clients 
        for being trained according to a federated policy 
        during current round.

        Returns
        -------
        list[Client]
            List of clients selected for training
        '''

        k = self._args.selected if self._args.selected else len(self._clients)
        return random.choices(self._clients, k = k)

class HybridSelection(ClientSelection):
    '''
    This strategy partitions clients into two groups of different size and
    different proabilities of being chosen for training at current round.

    Examples
    --------
    In this example training clients are partitioned into two random groups. The first
    of size `0.6 * len(clients)` will be assigned probability `0.3 / (0.6 * len(clients))`,
    whilst the second group of `(1 - 0.6) * len(clients)` will be assigned probability
    `(1 - 0.3) / ((1 - 0.6) * len(clients))`.

    >>> selection = HybridSelection(server, probability = 0.3, fraction = 0.6)
    >>> for round in range(rounds):
    >>>    clients = selection.select()
    >>>    ...
    '''

    def __init__(self, server: Any, probability: float = 0.5, fraction: float = 0.10):
        '''
        Constructs a hybrid strategy for client selection.

        Parameters
        ----------
        server: Server
            Reference to server instance
        probability: float
            Likelihood assigned to a `fraction` of clients
        fraction: float
            Fraction of total training clients which could be chosen
            with uniform `probability`
        '''

        super().__init__(server)
        # array of clients indices
        indices = np.arange(len(self._clients))
        # a 'fraction' of total clients have probability equal to 'probability' of being chosen
        indices_sharing_probability = random.choices(indices, k = math.floor(fraction * len(indices)))
        # remaining 1 - 'fraction' of total clients have probability 1 - 'probability' of being chosen
        self._importances[:] = (1 - probability) / (len(indices) - len(indices_sharing_probability))
        # 'fraction' of total clients have probability 'probability' of being selected, so it is assigned
        self._importances[indices_sharing_probability] = probability / len(indices_sharing_probability)

    def select(self) -> list[Any]:
        '''
        Selects with custom probabilities a bunch of clients 
        for being trained according to a federated policy 
        during current round.

        Returns
        -------
        list[Client]
            List of clients selected for training
        '''

        return random.choices(self._clients, k = self._args.selected, weights = self._importances)

class PowerOfChoiceSelection(ClientSelection):
    '''
    This elaborated strategy assigns higher likelihood of being selected
    to those clients showing a higher local loss in evalution mode.

    Notes
    -----
    A the beginning of the round this selection policy first chooses 
    `d` random clients according to their local datasets' sizes 
    (the larger the dataset, the higher the probability) and simply 
    evaluates the central model on their local dataset. Then `args.selected`
    with higher loss among the candidates are selected for training.
    '''

    def __init__(self, server: Any, d: int = 10):
        '''
        Constructs a power of choice strategy.

        Parameters
        ----------
        server: Server
            Reference to server instance
        d: int = 10
            Number of evaluated clients
        '''

        super().__init__(server)
        # number of evaluated candidates
        self.d = d

    def select(self) -> list[Any]:
        '''
        Selects with computed probabilities, according to
        power of choice strategy, a bunch of clients 
        for being trained according to a federated policy 
        during current round.

        Returns
        -------
        list[Client]
            List of clients selected for training
        '''

        # candidate clients are sampled according to their local datasets' sizes
        candidates = random.choices(
            range(len(self._clients)), 
            k = self.d,
            weights = [ len(client.dataset) for client in self._clients ]
        )
        # by default all clients have zero probabilities to be selected
        self._importances[:] = 0.0
        # no autograd needed for evaluation
        with torch.no_grad():
            for candidate in candidates:
                # just evaluate client's local loss
                loss, _, _ = self._clients[candidate].evaluate(self._model)
                # loss is used as sorting key in descending order
                self._importances[candidate] = loss
        # normalize proabilities
        self._importances /= self._importances.sum()
        # sort candidates according to probabilities and get indices or higher loss clients
        selected = np.argsort(self._importances)[-self._args.selected:]
        # now extracts training clients with higher loss
        return [ self._clients[index] for index in selected ]
