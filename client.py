import copy
import torch

from torch import optim, nn
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: nn.Module = model.to(self.device)
        # training set exists whether the client has been chosen as training client
        self.train_loader: DataLoader = None if test_client else DataLoader(
            self.dataset, 
            batch_size = self.args.bs, 
            shuffle = True, 
            drop_last = True
        )
        # validation set is passed in one unique big batch practically
        self.test_loader: DataLoader = DataLoader(
            self.dataset, 
            batch_size = len(self.dataset), 
            shuffle = False
        )
        # mean reduction over cross entropy across multiple batch samples
        self.criterion = nn.CrossEntropyLoss(ignore_index = 255, reduction = 'mean')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name

    def run_epoch(self, epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        for step, (images, labels) in enumerate(self.train_loader):
            # load data into appropriate device
            images, labels = images.to(self.device), labels.to(self.device)
            # compute predicted labels as logits
            predicted = self.model(images)
            # compute cross entropy loss
            loss = self.criterion(predicted, labels)
            # compute gradients
            optimizer.zero_grad()
            loss.backward()
            # clip gradient to avoid explosion
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
            # update weights with fgradients
            optimizer.step()


    def train(self, lr: float = None):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """

        # stochastic gradient descent optimizer
        optimizer: optim.Optimizer = optim.SGD(self.model.parameters(), lr = self.args.lr if lr is None else lr, momentum = self.args.m, weight_decay = self.args.m)
        # enable training mode
        self.model.train()
        # runs epochs of training
        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer)
        # length of dataset and model parameters
        return len(self.train_loader.dataset), self.model.state_dict()

    ###### USELESS USING EVALUATORS ######
    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        # enable validation
        self.model.eval()
        # run validation over each image
        with torch.no_grad():
            # get entire validation set, since it is limited in size
            images, labels = next(iter(self.test_loader))
            # load data into appropriate device
            images, labels = images.to(self.device), labels.to(self.device)
            # evaluate on test images
            outputs = self.model(images)
            # update score metrics
            metric.update(outputs, labels)
    #########################################

    ###### FIXME #######
    def loss(self, batched: bool = False) -> float:
        '''
        Computes local loss for power of method criterion.
        '''

        # enable validation
        self.model.eval()
        # total loss
        loss_ = 0
        # run validation over each image
        with torch.no_grad():
            images, labels = next(iter(DataLoader(
                self.dataset, 
                batch_size = min(32, len(self.dataset)) if batched else len(self.dataset), 
                shuffle = True
            )))
            # load data into appropriate device
            images, labels = images.to(self.device), labels.to(self.device)
            # evaluate on test images
            outputs = self.model(images)
            # update loss
            loss_ += self.criterion(outputs, labels)

        return loss_
    ###### FIXME #######
