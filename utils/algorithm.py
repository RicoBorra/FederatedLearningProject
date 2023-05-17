from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
from typing import Any, Iterable, Tuple

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

    def __init__(self, state: OrderedDict[str, torch.Tensor]):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        '''
        
        self._state = deepcopy(state)

    @property
    def state(self) -> OrderedDict[str, torch.Tensor]:
        '''
        Returns the state of central server.

        Notes
        -----
        After invoking any `visit(...)` and a final `aggregate(...)`,
        then the state changes.
        '''
        
        return self._state
    
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

    def __init__(self, state: OrderedDict[str, torch.Tensor]):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        '''

        super().__init__(state)
        
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
                lr = client.args.learning_rate, 
                momentum = client.args.momentum, 
                weight_decay = client.args.weight_decay
        )
        # train mode to enforce gradient computation
        model.train()
        # during local epochs the client model deviates
        # from the original configuration passed by the
        # central server
        for epoch in range(client.args.epochs):
            for x, y in client.loader:
                x = x.to(client.device)
                y = y.to(client.device)
                model.step(x, y, optimizer)

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

class FedProx(FedAvg):
    '''
    This algorithm is the fedprox, from Li et al. (2018), 
    which updates the central model by averaging clients model using their local datasets
    sizes as weights and uses a proximal term in optimization.
    '''

    def __init__(self, state: OrderedDict[str, torch.Tensor], mu: float = 1.0):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        mu: float
            Proximal term weight in local loss function (1.0 by default)
        '''

        super().__init__(state)
        
        self.mu = mu
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
                lr = client.args.learning_rate, 
                momentum = client.args.momentum, 
                weight_decay = client.args.weight_decay
        )
        # train mode to enforce gradient computation
        model.train()
        # during local epochs the client model deviates
        # from the original configuration passed by the
        # central server
        for epoch in range(client.args.epochs):
            for x, y in client.loader:
                x = x.to(client.device)
                y = y.to(client.device)
                # output logits of classes
                logits = model(x)
                # central model loss and reduction
                loss = model.criterion(logits, y)
                loss = model.reduction(loss, y)
                # add proximal term
                loss += FedProx.proxterm(mu = self.mu, weights = model.named_parameters(), state = self._state)
                # gradient computation and weights update
                optimizer.zero_grad()
                loss.backward()
                # clip gradient norm to avoid errors
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
