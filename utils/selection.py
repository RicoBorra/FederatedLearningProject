from abc import ABC, abstractmethod
import math
import numpy as np
import random
import torch
from typing import Any

# FIXME to be added to server logic
class ClientSelection(ABC):
    
    def __init__(self, server: Any):
        self._args = server.args
        self._clients = server.clients['training']
        self._model = server.model
        self._importances = np.ones(len(self._clients), dtype = np.float32)

    @property
    def probabilities(self) -> np.ndarray:
        return self._importances

    @abstractmethod
    def select(self) -> list[Any]:
        raise NotImplementedError()

class UniformSelection(ClientSelection):

    def select(self) -> list[Any]:
        return random.choices(self._clients, k = self._args.selected)

class HybridSelection(ClientSelection):

    def __init__(self, server: Any, probability: float = 0.5, fraction: float = 0.10):
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
        return random.choices(self._clients, k = self._args.selected, weights = self._importances)

class PowerOfChoiceSelection(ClientSelection):

    def __init__(self, server: Any, d: int = 10):
        super().__init__(server)

        self.d = d

    def select(self) -> list[Any]:
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
