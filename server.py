import abc
import copy
from collections import OrderedDict
import math
from typing import Iterable, Tuple

import numpy as np
import torch

import tqdm

import wandb

import client

class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.selection: ClientSelectionCriterion = None
        self.test_clients = test_clients
        self.model: torch.nn.Module = model
        self.metrics = metrics
        self.model_params_dict: OrderedDict = copy.deepcopy(self.model.state_dict())
        # assign selection criterion for both training and testing clients
        if self.args.selection == 'uniform':
            self.selection = UniformClientSelectionCriterion(
                n_clients = len(self.train_clients), 
                n_selected = min(self.args.clients_per_round, len(self.train_clients))
            )
        elif self.args.selection == 'hybrid':
            self.selection = HybridClientSelectionCriterion(
                n_clients = len(self.train_clients), 
                n_selected = min(self.args.clients_per_round, len(self.train_clients)),
                buckets = [(0.10, 0.5), (0.30, 0.0001)]
            )
        elif self.args.selection == 'poc':
            self.selection = PowerOfChoiceClientSelectionCriterion(
                n_clients = len(self.train_clients), 
                n_selected = min(self.args.clients_per_round, len(self.train_clients)),
                n_sampled = 100,
                clients = self.train_clients
            )

    def select_clients(self, train: bool = True):
        if train:
            num_clients = min(self.args.clients_per_round, len(self.train_clients))
            return np.random.choice(self.train_clients, num_clients, replace=False, p=self.selection.probabilities())
        
        num_clients = min(self.args.clients_per_round, len(self.test_clients))
        return np.random.choice(self.test_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, client in enumerate(clients):
            # model initialization in each round
            client.model.load_state_dict(self.model_params_dict)
            # train single client
            size, weights = client.train()
            # collect dataset size used for training and its trained coefficients
            updates.append((size, weights))
        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        
        # aggregated coeffients
        state = OrderedDict()
        # compute total dataset size over all clients
        total_size = sum([ size for size, _ in updates  ])
        # for each architecture's layer it aggregates the data
        # in a weighted average fashion
        for block in self.model_params_dict.keys():
            state[block] = torch.sum(
                torch.stack([ size / total_size * weights[block] for size, weights in updates ]), 
                dim = 0
            )
        # yields aggregated weights
        return state

    def train(self):
        """
        This method orchestrates the training, the evals and tests at rounds level
        """

        wandb.watch(self.model, log='all')

        wandb.define_metric('round')

        wandb.define_metric('accuracy/overall/training', step_metric='round')
        wandb.define_metric('accuracy/mean/training', step_metric='round')
        wandb.define_metric('accuracy/overall/testing', step_metric='round')
        wandb.define_metric('accuracy/mean/testing', step_metric='round')

        progress = tqdm.tqdm(total=self.args.num_rounds)
        progress.set_description('Training on federated devices')

        for r in range(self.args.num_rounds):
            # select clients for current round
            clients = self.select_clients()
            # collect updated models' weights from trained clients
            updates = self.train_round(clients)
            # aggregate updates from all clients
            self.model_params_dict = self.aggregate(updates)
            # update weights of centralized model
            self.model.load_state_dict(self.model_params_dict)
            # compute validation on training set
            if r % self.args.eval_interval == 0:
                self.eval_train()
                wandb.log({
                    'round': r + 1,
                    'accuracy/mean/training': self.metrics['eval_train'].results['Mean Acc'],
                    'accuracy/overall/training': self.metrics['eval_train'].results['Overall Acc']
                })
            # compute validation on validation set
            if r % self.args.test_interval == 0:
                self.test()
                wandb.log({
                    'round': r + 1,
                    'accuracy/mean/testing': self.metrics['test'].results['Mean Acc'],
                    'accuracy/overall/testing': self.metrics['test'].results['Overall Acc']
                })
            # print training metrics
            if r % self.args.print_train_interval == 0:
                print(self.metrics['eval_train'])
            # print validation metrics
            if r % self.args.print_test_interval == 0:
                print(self.metrics['test'])

            progress.update(1)

        progress.close()

        # last train evaluation update
        self.eval_train()
        wandb.log({
            'round': self.args.num_rounds,
            'accuracy/mean/training': self.metrics['eval_train'].results['Mean Acc'],
            'accuracy/overall/training': self.metrics['eval_train'].results['Overall Acc']
        })
        # last test evaluation update
        self.test()
        wandb.log({
            'round': self.args.num_rounds,
            'accuracy/mean/testing': self.metrics['test'].results['Mean Acc'],
            'accuracy/overall/testing': self.metrics['test'].results['Overall Acc']
        })

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        # select train clients
        clients = self.select_clients()
        # compute score metrics on each train client
        for client in clients:
            # model initialization in each round
            client.model.load_state_dict(self.model_params_dict)
            client.test(self.metrics['eval_train'])
        # compute training metrics
        self.metrics['eval_train'].get_results()

    def test(self):
        """
        This method handles the test on the test clients
        """
        # select test clients
        clients = self.select_clients(train=False)
        # compute score metrics on each train client
        for client in clients:
            # model initialization in each round
            client.model.load_state_dict(self.model_params_dict)
            client.test(self.metrics['test'])
        # compute test metrics
        self.metrics['test'].get_results()


class ClientSelectionCriterion(metaclass=abc.ABCMeta):

    def __init__(self, n_clients: int, n_selected: int):
        self.n_clients = n_clients
        self.n_selected = n_selected

    @abc.abstractproperty
    def probabilities(self) -> Iterable[float]:
        raise NotImplementedError()
    
class UniformClientSelectionCriterion(ClientSelectionCriterion):

    def __init__(self, n_clients: int, n_selected: int):
        super().__init__(n_clients, n_selected)
        self.probabilities_ = np.ones(n_clients, dtype = np.float32) / n_clients

    def probabilities(self) -> Iterable[float]:
        return self.probabilities_

class HybridClientSelectionCriterion(ClientSelectionCriterion):

    def __init__(self, n_clients: int, n_selected: int, buckets: Iterable[Tuple[float, float]], shuffle: bool = False):
        super().__init__(n_clients, n_selected)

        self.buckets = buckets
        self.shuffle = shuffle
        self.probabilities_ = np.ones(n_clients, dtype = np.float32)

        assert 0 < sum([ percent for percent, _ in buckets ]) <= 1

        cumulative_index = 0

        for percent, weight in buckets:
            n_clients_bucket = int(math.floor(percent * n_clients))
            self.probabilities_[cumulative_index:cumulative_index + n_clients_bucket] = weight
            cumulative_index += n_clients_bucket

        self.probabilities_ /= self.probabilities_.sum()
        
        if shuffle:
            np.random.shuffle(self.probabilities_)

    def probabilities(self) -> Iterable[float]:
        return self.probabilities_
    
class PowerOfChoiceClientSelectionCriterion(ClientSelectionCriterion):

    def __init__(self, n_clients: int, n_selected: int, clients: Iterable[client.Client], n_sampled: int = 100):
        super().__init__(n_clients, n_selected)
        self.clients = clients
        self.n_sampled = n_sampled
        self.indices_ = list(range(n_clients))
        self.probabilities_ = np.zeros(n_clients, dtype = np.float32)

    def probabilities(self) -> Iterable[float]:
        sampled = np.random.choice(self.indices_, self.n_sampled, replace = False)
        # compute local losses on train clients
        losses = [ (i, self.clients[i].loss(batched = True)) for i in sampled ]
        # sort clients by decreasing local loss
        losses.sort(key = lambda pair: pair[1], reverse = True)
        # selected indices
        selected = [ index for index, _ in losses[:self.n_selected] ]
        # assigns highest probability to K selectable clients
        self.probabilities_[:] = 0.0
        self.probabilities_[selected] = 1.0 / self.n_selected
        # yields updated probabilities
        return self.probabilities_
