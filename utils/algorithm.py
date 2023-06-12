from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Iterable, Tuple

from client import Client
from utils.reduction import MeanReduction

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

    def __init__(self, state: OrderedDict[str, torch.Tensor], scheduler: torch.optim.lr_scheduler.LRScheduler):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Learning rate scheduler over multiple server rounds
        '''
        
        self._state = deepcopy(state) if state is not None else None
        self._scheduler = scheduler

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
    
    @property
    def lr(self) -> float:
        '''
        Returns learning rate used by client at current round.

        Returns
        -------
        float
            Current learning rate
        '''

        return self._scheduler.get_last_lr()[0]

class FedAvg(FedAlgorithm):
    '''
    This algorithm is the plain federated averaging, from McMahan et al. (2017), 
    which updates the central model by averaging clients model using their local datasets
    sizes as weights.
    '''

    def __init__(self, state: OrderedDict[str, torch.Tensor], scheduler: torch.optim.lr_scheduler.LRScheduler):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Learning rate scheduler over multiple server rounds
        '''

        super().__init__(state, scheduler)
        
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
                lr = self.lr, 
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
                logits, loss = model.step(x, y, optimizer)

                ## FIXME checks
                # assert torch.isfinite(logits).all() and np.isfinite(loss)

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
        # aggregation means the end of current round, so we can update clients learning rate
        self._scheduler.optimizer.step()
        self._scheduler.step()

class FedProx(FedAvg):
    '''
    This algorithm is the fedprox, from Li et al. (2018), 
    which updates the central model by averaging clients model using their local datasets
    sizes as weights and uses a proximal term in optimization.
    '''

    def __init__(self, state: OrderedDict[str, torch.Tensor], scheduler: torch.optim.lr_scheduler.LRScheduler, mu: float = 1.0):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Learning rate scheduler over multiple server rounds
        mu: float
            Proximal term weight in local loss function (1.0 by default)
        '''

        super().__init__(state, scheduler)
        
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
                lr = self.lr, 
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

    def __init__(
            self, 
            state: OrderedDict[str, torch.Tensor],
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            beta_1: float = 0.9,
            beta_2: float = 0.99,
            tau: float = 1e-4,
            eta: float = 10 ** (-2.5)
        ):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Learning rate scheduler over multiple server rounds
        '''

        super().__init__(state, scheduler)
        
        self._updates: list[tuple[OrderedDict[str, torch.Tensor], int]] = []
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eta = eta
        self.tau = tau
        self.delta = OrderedDict({ key: torch.zeros(weight.size(), device = weight.device) for key, weight in state.items() })
        self.v = OrderedDict({ key: tau * tau * torch.ones(weight.size(), device = weight.device) for key, weight in state.items() })

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
            self.delta[key] = self.beta_1 * self.delta[key] + (1 - self.beta_1) * delta_t
            # squared delta_t for caching
            delta_t_squared = self.delta[key].square()
            # velocity update
            self.v[key] = self.v[key] - (1 - self.beta_2) * delta_t_squared * (self.v[key] - delta_t_squared).sign()
            # weights state update
            self._state[key] = self._state[key] + self.eta * self.delta[key] / (self.v[key].sqrt() + self.tau)
        # all updates have been consumed, so they can be removed
        self._updates.clear()
        # aggregation means the end of current round, so we can update clients learning rate
        self._scheduler.optimizer.step()
        self._scheduler.step()

class FedLeastSquares(FedAlgorithm):
    '''
    This algorithm implements federated least squares for the resolution of a ridge regression problem
    in a closed form.
    '''

    def __init__(
        self,
        state: OrderedDict[str, torch.Tensor],
        num_features: int,
        num_classes: int,
        lmbda: float,
        device: torch.device
    ):
        '''
        Initializes the algorithm with an initial state.

        Parameters
        ----------
        state: OrderedDict[str, torch.Tensor]
            State dictionary of parameters obtained from a model
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Learning rate scheduler over multiple server rounds
        '''

        super().__init__(state, None)
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.xtx = torch.zeros((num_features + 1, num_features + 1), device = device) 
        self.xty = torch.zeros((num_features + 1, num_classes), device = device)
        self.device = device
        self.lmbda = lmbda

        self._n_samples = 0

    def visit(self, client: Client, model: nn.Module = None) -> Any:
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

        self._n_samples += len(client.dataset)
        
        # loads clients data
        loader = torch.utils.data.DataLoader(client.dataset, batch_size = len(client.dataset))
        x, y = next(iter(loader))
        # appends one column (for bias) to features matrix
        ones = torch.ones((x.shape[0], 1))
        x = torch.cat((ones, x), dim = -1).to(self.device)
        # center class label in range [-1, 1]
        y = (torch.nn.functional.one_hot(y, num_classes = self.num_classes) * 2) - 1
        y = y.type(torch.float32).to(self.device)
        # computes aggregated matrix as updates
        xtx = x.T @ x
        xty = x.T @ y
        # online aggregation
        self.xtx += xtx
        self.xty += xty
        # returns it to outside
        return xtx, xty

    def aggregate(self):
        '''
        Updates central model state by computing a weighted average of clients' states with their local datasets' sizes.
        '''

        # computes closed form of ridge regression
        regularized_xtx = self.xtx + self.lmbda * torch.eye(n = self.num_features + 1, device = self.device)
        self._state['beta'] = torch.linalg.solve(regularized_xtx, self.xty)

class FedAvgSVRG(FedAvg):

    def __init__(self, state: OrderedDict[str, torch.Tensor]):
        super().__init__(state, None)

    def visit(self, client: Client, model: nn.Module) -> Any:
        
        weight_estimate = self._state
        gradient_estimates = [ None for _ in model.parameters() ]
        loss_estimates = [ None for _ in model.parameters() ]
        lr = client.args.learning_rate
        local_weight = deepcopy(self._state)
        running_avg = deepcopy(local_weight)
        running_size = 0
        weight_decay = client.args.weight_decay
        
        if isinstance(next(iter(model.parameters())), nn.Parameter) == False:
            raise NotImplementedError()

        # train mode to enforce gradient computation
        model.train()
        # during local epochs the client model deviates
        # from the original configuration passed by the
        # central server

        for epoch in range(client.args.epochs):

            # compute weight estimate and gradient estimate
            if epoch % 2 == 0:
                weight_estimate = running_avg
                running_avg = deepcopy(local_weight)
                running_size = 0
                # FIXME losses = []
                # FIXME sizes = []
                total_loss = 0.0
                total_size = 0

                model.load_state_dict(weight_estimate)

                for x, y in client.loader:
                    x = x.to(client.device)
                    y = y.to(client.device)
                    # model.load_state_dict(weight_estimate)
                    logits = model(x)
                    # central model loss and reduction
                    loss = model.criterion(logits, y)
                    # loss = model.reduction(loss, y)
                    loss = loss.mean()
                    size = y.shape[0]
                    # FIXME losses.append(loss * size)
                    # FIXME sizes.append(size)
                    total_loss += loss * size
                    total_size += size

                # FIXME total_loss = sum(losses) / sum(sizes)
                total_loss = total_loss / total_size
                model.zero_grad()
                total_loss.backward()

                for i, p in enumerate(model.parameters()):
                    gradient_estimates[i] = p.grad.detach()
    
            # do the regular training protocol
            for x, y in client.loader:
                x = x.to(client.device)
                y = y.to(client.device)

                # needs to be computed first so we load local weight only once
                model.load_state_dict(weight_estimate)
                logits = model(x)
                # loss of the best estimate
                loss2 = model.criterion(logits, y)
                loss2 = loss2.mean()

                model.zero_grad()
                loss2.backward()
                for i, p in enumerate(model.parameters()):
                    loss_estimates[i] = p.grad.detach()


                model.load_state_dict(local_weight)
                logits = model(x)
                # central model loss 
                loss1 = model.criterion(logits, y)
                loss1 = loss1.mean()
            
                l2penalty = weight_decay * sum([p.norm() for p in model.parameters()])

                loss = loss1 + weight_decay * l2penalty
                model.zero_grad()
                loss.backward()

                # SVRG update rule
                for i, p in enumerate(model.parameters()):
                    if p.grad is None:
                        continue
                    p = p - lr * (p.grad - loss_estimates[i] + gradient_estimates[i])

                local_weight = model.state_dict()
                # compute runnning average of model weights, to be used as estimate
                for run, cur in zip(running_avg.values(), local_weight.values()):
                    run = (run * running_size + cur) / (running_size + 1)
                running_size += 1
                ## FIXME checks
                # assert torch.isfinite(logits).all() and np.isfinite(loss)

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
        # aggregation means the end of current round, so we can update clients learning rate
        

class FedAvgTransformedSVRG(FedAvg):

    def __init__(self, state: OrderedDict[str, torch.Tensor]):
        super().__init__(state, None)

    def visit(self, client: Client, model: nn.Module) -> Any:

        weight_estimate = self._state
        gradient_estimates = [ None for _ in model.parameters() ]
        loss_estimates = [ None for _ in model.parameters() ]
        lr = client.args.learning_rate
        local_weight = deepcopy(self._state)
        running_avg = deepcopy(local_weight)
        running_size = 0
        weight_decay = client.args.weight_decay
        
        if isinstance(next(iter(model.parameters())), nn.Parameter) == False:
            raise NotImplementedError()

        # train mode to enforce gradient computation
        model.train()
        # during local epochs the client model deviates
        # from the original configuration passed by the
        # central server

        for epoch in range(client.args.epochs):

            # compute weight estimate and gradient estimate
            if epoch % 2 == 0:
                weight_estimate = running_avg
                running_avg = deepcopy(local_weight)
                running_size = 0
                # FIXME losses = []
                # FIXME sizes = []
                total_loss = 0.0
                total_size = 0

                model.load_state_dict(weight_estimate)

                for x, y in client.loader:
                    x = x.to(client.device)
                    y = y.to(client.device)
                    # appends one column (for bias) to features matrix
                    ones = torch.ones((x.shape[0], 1), device = client.device)
                    x = torch.cat((ones, x), dim = -1)
                    # center class label in range [-1, 1]
                    y_binarized = (torch.nn.functional.one_hot(y, num_classes = 62) * 2) - 1
                    y_binarized = y_binarized.type(torch.float32)
                    # model.load_state_dict(weight_estimate)
                    logits = model(x)
                    # central model loss and reduction
                    loss = model.criterion(logits, y_binarized)
                    # loss = model.reduction(loss, y)
                    loss = loss.mean()
                    size = y.shape[0]
                    # FIXME losses.append(loss * size)
                    # FIXME sizes.append(size)
                    total_loss += loss * size
                    total_size += size

                # FIXME total_loss = sum(losses) / sum(sizes)
                total_loss = total_loss / total_size
                model.zero_grad()
                total_loss.backward()

                for i, p in enumerate(model.parameters()):
                    gradient_estimates[i] = p.grad.detach()
    
            # do the regular training protocol
            for x, y in client.loader:
                x = x.to(client.device)
                y = y.to(client.device)
                # appends one column (for bias) to features matrix
                ones = torch.ones((x.shape[0], 1), device = client.device)
                x = torch.cat((ones, x), dim = -1)
                # center class label in range [-1, 1]
                y_binarized = (torch.nn.functional.one_hot(y, num_classes = 62) * 2) - 1
                y_binarized = y_binarized.type(torch.float32)
                # needs to be computed first so we load local weight only once
                model.load_state_dict(weight_estimate)
                logits = model(x)
                # loss of the best estimate
                loss2 = model.criterion(logits, y_binarized)
                loss2 = loss2.mean()

                model.zero_grad()
                loss2.backward()
                for i, p in enumerate(model.parameters()):
                    loss_estimates[i] = p.grad.detach()


                model.load_state_dict(local_weight)
                logits = model(x)
                # central model loss 
                loss1 = model.criterion(logits, y_binarized)
                loss1 = loss1.mean()
            
                l2penalty = weight_decay * sum([p.norm() for p in model.parameters()])

                loss = loss1 + weight_decay * l2penalty
                model.zero_grad()
                loss.backward()

                # SVRG update rule
                for i, p in enumerate(model.parameters()):
                    if p.grad is None:
                        continue
                    p = p - lr * (p.grad - loss_estimates[i] + gradient_estimates[i])

                local_weight = model.state_dict()
                # compute runnning average of model weights, to be used as estimate
                for run, cur in zip(running_avg.values(), local_weight.values()):
                    run = (run * running_size + cur) / (running_size + 1)
                running_size += 1
                ## FIXME checks
                # assert torch.isfinite(logits).all() and np.isfinite(loss)

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
        # aggregation means the end of current round, so we can update clients learning rate