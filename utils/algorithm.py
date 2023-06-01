from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Iterable, Tuple

from .configuration import AlgorithmConfiguration
from client import Client

class FedAlgorithm(ABC):
    '''
    Base class for any federated algorithm used for aggregating
    partial updates from clients and update the central state of
    the server.

    Notes
    -----
    Deep copy is invoked implicitly through `copy.deepcopy`
    on state dictionaries obtained from models, in order to avoid
    undesired changes on parameters.

    Examples
    --------
    This example explains the main logic behind.

    >>> model = CNN(...)
    >>> initial_state = model.state_dict()
    >>> algorithm = FedAlgorithm(initial_state)
    >>> ...
    >>> for client in range(n_clients):
    >>>     # train central model with initial round state
    >>>     # the algorithm handles update collection and all
    >>>     update = algorithm.visit(client, model)
    >>> ...
    >>> # the algorithm updates the internal state and returns it
    >>> algorithm.aggregate()
    >>> updated_state = algorithm.state
    '''

    def __init__(self, state: OrderedDict[str, torch.Tensor], configuration: AlgorithmConfiguration):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        configuration: AlgorithmConfiguration
            Hyperparameters configuration of the algorithm
        '''
        
        self._state = deepcopy(state)
        self._configuration = configuration

    @property
    def state(self) -> OrderedDict[str, torch.Tensor]:
        '''
        Returns the state of central server.

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Pytorch state dict of model

        Notes
        -----
        After invoking any `visit(...)` and a final `aggregate(...)`,
        then the state changes.
        '''
        
        return self._state
    
    @property
    def configuration(self) -> AlgorithmConfiguration:
        '''
        Returns the configuration of central server.

        Returns
        -------
        AlgorithmConfiguration
            Algorithm configuration dictionary
        '''
        
        return self._configuration
    
    @abstractmethod
    def visit(self, client: Client, model: nn.Module) -> Any:
        '''
        Visits single client and runs federated algorithm on it, returning local update.

        Parameters
        ----------
        client: Client
            Terminal client on which the algorithm is executed
        model: nn.Module
            Central server model (by reference !!) to be locally trained on the client

        Returns
        -------
        Any
            Collected update by the algorithm
        '''

        raise NotImplementedError()

    @abstractmethod
    def aggregate(self):
        '''
        Updates server model (central) state aggregating all clients updates.
        '''

        raise NotImplementedError()

class FedAvg(FedAlgorithm):
    '''
    This algorithm is the plain federated averaging, from McMahan et al. (2017), 
    which updates the central model by averaging clients model using their local datasets
    sizes as weights.
    '''

    def __init__(self, state: OrderedDict[str, torch.Tensor], configuration: AlgorithmConfiguration):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        configuration: AlgorithmConfiguration
            Hyperparameters configuration of the algorithm
        '''

        super().__init__(state, configuration)
        
        self._updates: list[tuple[OrderedDict[str, torch.Tensor], int]] = []

    def visit(self, client: Client, model: nn.Module) -> Any:
        '''
        Visits single client and runs federated algorithm on it, returning local update.

        Parameters
        ----------
        client: Client
            Terminal client on which the algorithm is executed
        model: nn.Module
            Central server model (by reference !!) to be locally trained on the client

        Returns
        -------
        tuple[OrderedDict[str, torch.Tensor], int]
            Update which should be explicitly collectd by `accumulate(...)`

        Notes
        -----
        The update consists in a tuple of client trained weights and local dataset size.
        '''
        
        # first the model is passed by reference to the client just to speed up 
        # the computation, yet the model is initialized with the same parameters of 
        # the central model at the beginning of the round
        model.load_state_dict(self._state)
        # plain stochastic gradient descent
        # weight decay is L2 (ridge) penalty
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr = self._configuration.lr, 
            momentum = self._configuration.momentum, 
            weight_decay = self._configuration.weight_decay
        )
        # train mode to enforce gradient computation
        print(self._configuration)
        model.train()
        # during local epochs the client model deviates
        # from the original configuration passed by the
        # central server
        for epoch in range(int(self._configuration.epochs)):
            for x, y in client.loader:
                x = x.to(client.device)
                y = y.to(client.device)
                logits, loss = model.step(x, y, optimizer)
        # clone of updated client parameters and size of local dataset
        update = deepcopy(model.state_dict()), len(client.dataset)
        # appended to round history of updates
        self._updates.append(update)
        # returns it to outside
        return update

    def aggregate(self):
        '''
        Updates central model state by computing a weighted average of clients' states with their local datasets' sizes.
        '''
        
        # total size is the sum of the sizes from all contributing clients' datasets
        n = sum([ size for _, size in self._updates ])
        # for each architectural part of the state (e.g. 'fc.weight' or 'conv.bias'), the central state is obtained
        # by averaging all clients state of the same part
        for key in self._state.keys():
            # detach is invoked to enforce the detachment from autograd graph when making computations on the resulting
            # object
            self._state[key] = torch.sum(
                torch.stack([ size / n * weights[key].detach() for weights, size in self._updates ]),
                dim = 0
            )
        # all updates have been consumed, so they can be removed
        self._updates.clear()
        # aggregation means the end of current round, so we can update parameters eventually
        self._configuration.update()

class FedProx(FedAvg):
    '''
    This algorithm is the fedprox, from Li et al. (2018), 
    which updates the central model by averaging clients model using their local datasets
    sizes as weights and uses a proximal term in optimization.
    '''

    def __init__(self, state: OrderedDict[str, torch.Tensor], configuration: AlgorithmConfiguration):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        configuration: AlgorithmConfiguration
            Hyperparameters configuration of the algorithm
        '''

        super().__init__(state, configuration)
        
        self._updates: list[tuple[OrderedDict[str, torch.Tensor], int]] = []

    def visit(self, client: Client, model: nn.Module) -> Any:
        '''
        Visits single client and runs federated algorithm on it, returning local update.

        Parameters
        ----------
        client: Client
            Terminal client on which the algorithm is executed
        model: nn.Module
            Central server model (by reference !!) to be locally trained on the client

        Returns
        -------
        tuple[OrderedDict[str, torch.Tensor], int]
            Update which should be explicitly collectd by `accumulate(...)`

        Notes
        -----
        The update consists in a tuple of client trained weights and local dataset size.
        '''

        # first the model is passed by reference to the client just to speed up 
        # the computation, yet the model is initialized with the same parameters of 
        # the central model at the beginning of the round
        model.load_state_dict(self._state)
        # plain stochastic gradient descent
        # weight decay is L2 (ridge) penalty
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr = self._configuration.lr, 
            momentum = self._configuration.momentum, 
            weight_decay = self._configuration.weight_decay
        )
        # train mode to enforce gradient computation
        print(self._configuration)
        model.train()
        # during local epochs the client model deviates
        # from the original configuration passed by the
        # central server
        for epoch in range(int(self._configuration.epochs)):
            for x, y in client.loader:
                x = x.to(client.device)
                y = y.to(client.device)
                # output logits of classes
                logits = model(x)
                # central model loss and reduction
                loss = model.criterion(logits, y)
                loss = model.reduction(loss, y)
                # add proximal term
                loss += FedProx.proxterm(mu = self._configuration.mu, weights = model.named_parameters(), state = self._state)
                # gradient computation and weights update
                optimizer.zero_grad()
                loss.backward()
                # clip gradient norm to avoid errors
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                # update weights
                optimizer.step()
        # clone of updated client parameters and size of local dataset
        update = deepcopy(model.state_dict()), len(client.dataset)
        # appended to round history of updates
        self._updates.append(update)
        # returns it to outside
        return update
    
    @staticmethod
    def proxterm(mu: float, weights: Iterable[Tuple[str, torch.nn.Parameter]], state: OrderedDict[str, torch.Tensor]) -> float:
        '''
        Computes proximal term.

        Parameters
        ----------
        mu: float
            Coefficient of proximal term
        weights: Iterable[Tuple[str, torch.nn.Parameter]]
            Weights of drifted model updated locally with gradient graph attached
        state: OrderedDict[str, torch.Tensor]
            Original fixed state of central server

        Returns
        -------
        float
            Proximal term
        '''

        term = 0.0
        # for each architectural part of the state (e.g. 'fc.weight' or 'conv.bias'), add squared L2 norm of drift
        for name, weight in weights:
            # detach is invoked to enforce the detachment from autograd graph
            term += (weight - state[name].detach()).square().sum()
        # multiply using mu lagrangian
        return 0.5 * mu * term

class FedYogi(FedAvg):
    '''
    This algorithm converges faster than plain vanilla FedAvg by exploiting
    server learning rate and momentum when doing clients' updates aggregation.
    Particularly, server side, FedYogi is adopted as optimizer.
    '''

    def __init__(self, state: OrderedDict[str, torch.Tensor], configuration: AlgorithmConfiguration):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        configuration: AlgorithmConfiguration
            Hyperparameters configuration of the algorithm
        '''

        super().__init__(state, configuration)
        
        self._updates: list[tuple[OrderedDict[str, torch.Tensor], int]] = []
        self.delta = OrderedDict({ key: torch.zeros(weight.size(), device = weight.device) for key, weight in state.items() })
        self.v = OrderedDict({ key: self._configuration.tau * self._configuration.tau * torch.ones(weight.size(), device = weight.device) for key, weight in state.items() })

    def aggregate(self):
        '''
        Updates central model state by computing a weighted average of clients' states with their local datasets' sizes
        and server learning rate.
        '''

        # total size is the sum of the sizes from all contributing clients' datasets
        n = sum([ size for _, size in self._updates ])
        # for each architectural part of the state (e.g. 'fc.weight' or 'conv.bias'), the central state is obtained
        # by averaging all clients state of the same part
        # current delta given by differences of local weights with respect to intial round weights (state)
        for key in self._state.keys():
            # detach is invoked to enforce the detachment from autograd graph when making computations on the resulting object
            delta_t = torch.sum(
                torch.stack([ size / n * (weights[key].detach() - self._state[key]) for weights, size in self._updates ]),
                dim = 0
            )
            # updates server delta
            self.delta[key] = self._configuration.beta_1 * self.delta[key] + (1 - self._configuration.beta_1) * delta_t
            # squared delta_t for caching
            delta_t_squared = self.delta[key].square()
            # velocity update
            self.v[key] = self.v[key] - (1 - self._configuration.beta_2) * delta_t_squared * (self.v[key] - delta_t_squared).sign()
            # weights state update
            self._state[key] = self._state[key] + self._configuration.eta * self.delta[key] / (self.v[key].sqrt() + self._configuration.tau)
        # all updates have been consumed, so they can be removed
        self._updates.clear()
        # aggregation means the end of current round, so we can update parameters eventually
        print(self._configuration)
        self._configuration.update()

class FedSr(FedAvg):
    '''
    This algorithm is FedSr domain generalization algorithm in federated scenario.
    '''

    def visit(self, client: Client, model: nn.Module) -> Any:
        '''
        Visits single client and runs federated algorithm on it, returning local update.

        Parameters
        ----------
        client: Client
            Terminal client on which the algorithm is executed
        model: nn.Module
            Central server model (by reference !!) to be locally trained on the client

        Returns
        -------
        tuple[OrderedDict[str, torch.Tensor], int]
            Update which should be explicitly collectd by `accumulate(...)`

        Notes
        -----
        The update consists in a tuple of client trained weights and local dataset size.
        '''
        
        # loads (decayed ?) parameters of fedsr to the model which is expected to have 'beta_kl' and 'beta_l2' attributes
        model.beta_l2 = self._configuration.beta_l2
        model.beta_kl = self._configuration.beta_kl
        print(self._configuration)
        # executes normal fedavg
        return super().visit(client, model)