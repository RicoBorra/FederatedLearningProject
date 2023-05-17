import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Any
import wandb

from client import Client
from utils.evaluation import FederatedMetrics

class Server(object):
    '''
    This simulates a central server running a federated learning
    algorithm to train a model across terminal clients.
    '''

    def __init__(
        self,
        algorithm: type,
        model: nn.Module,
        clients: dict[str, Client],
        args: Any
    ):
        '''
        Initialized a server training a central model on terminal clients in a federated setting.

        Parameters
        -----------
        algorithm: type
            Class type of the algorithm to be used, see `FedAlgorithm` or `FedAvg`
        model: nn.Module
            Central model which is passed by reference across clients for the sake of the simulation
        clients: dict[str, Client]
            Groups of clients, divided among `training`, `validation` and `testing`
        args: Any
            Command line arguments of the simulation
        '''
        
        self.args = args
        self.model = model
        self.algorithm = algorithm(model.state_dict())
        self.clients = clients
        self.evaluators = { group: FederatedMetrics() for group in clients.keys() }
    
    def select(self, group: str = 'training') -> list[Client]:
        '''
        Randomly sample (no replacement) `args.selected` clients from `group` to be trained.

        Parameters
        ----------
        group: str
            Group of clients from which to sample (training by default)

        Returns
        -------
        list[Client]
            List of sampled clients
        '''
        
        return random.sample(self.clients[group], k = self.args.selected)

    def run(self):
        '''
        Trains the central model for `args.rounds` rounds on clients from
        `training` group, of which `args.selected` are randomly sampled each
        round to be trained locally for `args.epochs`.
        '''

        # progress bar for training 'args.selected' local clients at each round
        locals = tqdm(total = self.args.selected, desc = '[+] local models training')
        # logs both to screen and remotely
        self.initialize_logger()
        # runs 'args.rounds' rounds of central training
        for round in tqdm(range(self.args.rounds), desc = '[+] central server training'):
            # progress bar is reinitialized to zero
            locals.reset()
            # selects 'args.selected' training clients and trains them
            for client in self.select():
                # the local client update is returned and collected by the federated algorithm
                client.train(self.algorithm, self.model)
                # progress bar update in 1...args.selected
                locals.update(1)
            # disable gradient computations during evaluation
            with torch.no_grad():
                # finally, at the end of the round, the new model
                # state parameters are updated by aggregating local
                # updates from client
                self.algorithm.aggregate()
                # central model parameters are reinitialized with those
                # computed with the federated learning algorithm
                self.model.load_state_dict(self.algorithm.state)
                # eventually evaluates the model on training and validation cloents
                if round % self.args.evaluation == 0:
                    # FIXME subset evaluation
                    self.evaluate(round, fraction = self.args.evaluation_fraction)
                # eventually saves updated central model parameters as checkpoint
                if self.args.checkpoint is not None and round > 0 and round % int(self.args.checkpoint[0]) == 0:
                    self.save(round)
        # final evaluation
        with torch.no_grad():
            self.evaluate(self.args.rounds, fraction = self.args.evaluation_fraction)

    def evaluate(self, round: int, fraction: float):
        '''
        Evaluates performance of central model on `training` and `validation` clients and logs
        performance metrics.

        Parameters
        ----------
        round: int
            Training round of the server
        fraction: float
            Fraction of clients from each group to be evaluated (1.0 by default)  
        '''

        # FIXME subset evaluation
        validators = random.sample(self.clients['training'], k = math.floor(fraction * len(self.clients['training']))) if fraction < 1.0 else self.clients['training']
        # evaluation of clients from 'training' group
        for client in tqdm(validators, desc = '[+] evaluating training clients'):
            client.validate(self.model, self.evaluators['training'])
        # FIXME subset evaluation
        validators = random.sample(self.clients['validation'], k = math.floor(fraction * len(self.clients['validation']))) if fraction < 1.0 else self.clients['validation']
        # evaluation of clients from 'validation' group
        for client in tqdm(validators, desc = '[+] evaluating validation clients'):
            client.validate(self.model, self.evaluators['validation'])
        # compute metrics at the end of the round
        self.evaluators['training'].compute()
        self.evaluators['validation'].compute()
        # log metrics on screen
        print(
            f"[+] accuracy: {100 * self.evaluators['training']['accuracy']:.3f}%, "
            f"weighted accuracy: {100 * self.evaluators['training']['weighted_accuracy']:.3f}% (training)\n"
            f"[+] accuracy: {100 * self.evaluators['validation']['accuracy']:.3f}%, "
            f"weighted accuracy: {100 * self.evaluators['validation']['weighted_accuracy']:.3f}% (validation)"
        )
        # log metrics remotely to weights and biases
        wandb.log({
            'round': round + 1,
            'accuracy/weighted/training': self.evaluators['training']['weighted_accuracy'],
            'accuracy/overall/training': self.evaluators['training']['accuracy']
        })
        wandb.log({
            'round': round + 1,
            'accuracy/weighted/validation': self.evaluators['validation']['weighted_accuracy'],
            'accuracy/overall/validation': self.evaluators['validation']['accuracy']
        })

    def save(self, round: int):
        '''
        Saves the central model parameters as a checkpoint.

        Parameters
        ----------
        round: int
            Training round of the server
        '''

        # model name is a combination of the round and hash of its parameters
        name = f'round_{str(round).zfill(4)}_hash_{hash(str(self.algorithm.state))}.pt'
        torch.save(self.algorithm.state, os.path.join(self.args.checkpoint[1], name))

    def initialize_logger(self):
        '''
        Initializes remote logger on Weights & Biases.

        Notes
        -----
        Logs model gradients, weights and the class weighted and mean accuracy
        both for training and validation clients.
        '''

        # this loads models' parameters and gradients at each log
        wandb.watch(self.model, log='all')
        # this uses the round as x-axis metric when representing curves
        wandb.define_metric('round')
        # represented curves are training and validation accuracies (class weighted and overall)
        wandb.define_metric('accuracy/overall/training', step_metric='round')
        wandb.define_metric('accuracy/weighted/training', step_metric='round')
        wandb.define_metric('accuracy/overall/validation', step_metric='round')
        wandb.define_metric('accuracy/weighted/validation', step_metric='round')