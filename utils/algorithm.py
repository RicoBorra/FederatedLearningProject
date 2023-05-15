from abc import ABC
from collections import OrderedDict
import torch
from typing import Any

class FederatedAlgorithm(ABC):
    '''
    Base class for any federated algorithm used for aggregating
    partial updates from clients and update the central state of
    the server.

    Notes
    -----
    Deep copy must be invoked explicitly through `copy.deepcopy`
    on state dictionaries obtained from models, in order to avoid
    undesired changes on parameters.

    Examples
    --------
    This example explains the main logic behind.

    >>> model = CNN(...)
    >>> initial_state = deepcopy(model.state_dict())
    >>> algorithm = FederatedAlgorithm(initial_state)
    >>> ...
    >>> for client in range(n_clients):
    >>>     # train client model with initial state
    >>>     model.load_state_dict(algorithm.state)
    >>>     client_update = train(model)
    >>>     # collect client update
    >>>     algorithm.accumulate(client_update)
    >>> ...
    >>> # the algorithm updates the internal state and returns it
    >>> algorithm.update()
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
        
        self._state = state

    def accumulate(self, update: Any):
        '''
        Collects client update.

        Parameters
        ----------
        update: Any
            Update object obtained from a client
        '''

        raise NotImplementedError()

    def update(self):
        '''
        Updates server model (central) state consuming all clients updates.
        '''

        raise NotImplementedError()

    @property
    def state(self) -> OrderedDict[str, torch.Tensor]:
        '''
        Returns the state of central server.

        Notes
        -----
        After invoking any `collect(...)` and a final `update(...)`,
        then the state changes.
        '''
        
        return self._state

class FederatedAverage(FederatedAlgorithm):
    '''
    This algorithm is the plain federated averaging which updates
    the central model by averaging clients model using their local datasets
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

    def accumulate(self, update: tuple[OrderedDict[str, torch.Tensor], int]):
        '''
        Collects client update which consists of client local state and its local dataset size.

        Parameters
        ----------
        update: tuple[OrderedDict[str, torch.Tensor], int]
            Tuple of client local state and its local dataset size after local training epochs
        '''

        self._updates.append(update)

    def update(self):
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
