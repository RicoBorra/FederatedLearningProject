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
from utils.selection import UniformSelection, HybridSelection, PowerOfChoiceSelection
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
        # appropriate client selection strategy
        self.initialize_client_selection_strategy()
        

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
            for client in self.selection.select():
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
            # eventually involves clients from `testing`
            self.evaluate(
                self.args.rounds, 
                fraction = self.args.evaluation_fraction, 
                testing = getattr(self.args, 'testing', False)
            )

    def evaluate(self, round: int, fraction: float, testing: bool = False):
        '''
        Evaluates performance of central model on `training` and `validation` clients and logs
        performance metrics.

        Parameters
        ----------
        round: int
            Training round of the server
        fraction: float
            Fraction of clients from each group to be evaluated (1.0 by default)
        testing: bool
            Tells whether to run evaluation on the original hold out clients of
            `testing` group (False by default)
        '''

        # FIXME subset evaluation
        # FIXME remove 50
        validators = random.sample(self.clients['training'], k = math.floor(fraction * len(self.clients['training']))) if fraction < 1.0 else self.clients['training']
        # validators = random.sample(self.clients['training'], k = 100)
        # evaluation of clients from 'training' group
        for client in tqdm(validators, desc = '[+] evaluating training clients'):
            client.validate(self.model, self.evaluators['training'])
        # FIXME subset evaluation
        # FIXME remove 50
        validators = random.sample(self.clients['validation'], k = math.floor(fraction * len(self.clients['validation']))) if fraction < 1.0 else self.clients['validation']
        # validators = random.sample(self.clients['validation'], k = 100)
        # evaluation of clients from 'validation' group
        for client in tqdm(validators, desc = '[+] evaluating validation clients'):
            client.validate(self.model, self.evaluators['validation'])
        # compute metrics at the end of the round
        self.evaluators['training'].compute()
        self.evaluators['validation'].compute()
        # log metrics on screen
        print(
            # training metrics
            f"[+] accuracy: {100 * self.evaluators['training']['accuracy']:.3f}%, "
            f"weighted accuracy: {100 * self.evaluators['training']['weighted_accuracy']:.3f}%, "
            f"loss: {self.evaluators['training']['ce_loss']:.5f} (training)\n"
            # validation metrics
            f"[+] accuracy: {100 * self.evaluators['validation']['accuracy']:.3f}%, "
            f"weighted accuracy: {100 * self.evaluators['validation']['weighted_accuracy']:.3f}%, "
            f"loss: {self.evaluators['validation']['ce_loss']:.5f} (validation)"
        )
        # log metrics remotely to weights and biases
        wandb.log({
            'round': round + 1,
            'accuracy/weighted/training': self.evaluators['training']['weighted_accuracy'],
            'accuracy/overall/training': self.evaluators['training']['accuracy'],
            'loss/training': self.evaluators['training']['ce_loss']
        })
        wandb.log({
            'round': round + 1,
            'accuracy/weighted/validation': self.evaluators['validation']['weighted_accuracy'],
            'accuracy/overall/validation': self.evaluators['validation']['accuracy'],
            'loss/validation': self.evaluators['validation']['ce_loss']
        })
        # eventually this is executed at the end of the entire simulation in order to
        # understand how well the central model performs on new unseen `testing` clients
        # completely disjoint with respect to train `training` and `validation` clients
        # NOTE all clients, not just a fraction, from `testing` group are evaluated
        if testing:
            print('[+] final evaluation on all unseen testing clients', end = '')
            # evaluation of clients from 'testing' group
            for client in tqdm(self.clients['testing'], desc = '[+] evaluating testing clients'):
                client.validate(self.model, self.evaluators['testing'])
            # compute metrics at the end of the round
            self.evaluators['testing'].compute()
            # log metrics on screen
            print(
                # testing metrics
                f"[+] accuracy: {100 * self.evaluators['testing']['accuracy']:.3f}%, "
                f"weighted accuracy: {100 * self.evaluators['testing']['weighted_accuracy']:.3f}%, "
                f"loss: {self.evaluators['testing']['ce_loss']:.5f} (validation)"
            )
            # log metrics remotely to weights and biases
            wandb.log({
                'round': round + 1,
                'accuracy/weighted/testing': self.evaluators['testing']['weighted_accuracy'],
                'accuracy/overall/testing': self.evaluators['testing']['accuracy'],
                'loss/testing': self.evaluators['testing']['ce_loss']
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
        wandb.watch(self.model, log = 'all')
        # this uses the round as x-axis metric when representing curves
        wandb.define_metric('round')
        # represented curves are training and validation accuracies (class weighted and overall) and losses
        wandb.define_metric('accuracy/overall/training', step_metric = 'round')
        wandb.define_metric('accuracy/weighted/training', step_metric = 'round')
        wandb.define_metric('accuracy/overall/validation', step_metric = 'round')
        wandb.define_metric('accuracy/weighted/validation', step_metric = 'round')
        wandb.define_metric('loss/training', step_metric = 'round')
        wandb.define_metric('loss/validation', step_metric = 'round')

    def initialize_client_selection_strategy(self):
        '''
        Initializes client selection strategy according to simulation parameters.
        '''

        # uniform selection is default, all training clients are equally likely of being selected each round
        if not self.args.selection or self.args.selection[0] == 'uniform':
            self.selection = UniformSelection(server = self)
        # a fraction of clients shares a certain probability, while remainng clients have the left probability
        elif self.args.selection[0] == 'hybrid':
            if len(self.args.selection) < 3:
                self.selection = HybridSelection(server = self)
            else:
                self.selection = HybridSelection(server = self, probability = float(self.args.selection[1]), fraction = float(self.args.selection[2]))
        # power of choice favors clients with a higher local loss among those with larger datasets
        elif self.args.selection[0] == 'poc':
            if len(self.args.selection) < 2:
                self.selection = PowerOfChoiceSelection(server = self)
            else:
                self.selection = PowerOfChoiceSelection(server = self, d = int(self.args.selection[1]))
        # invalid client selection strategy
        else:
            raise RuntimeError(f'unrecognized selection strategy \'{self.args.selection[0]}\', expected \'uniform\', \'hybrid\' or \'poc\'')