import pdb
import torch
import numpy as np
import torch.optim as optim
from .cosinelr import CosineAnnealingWarm

class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                momentum=0.1,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            alpha = self.optim_dict['learning_ratio']
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer)

    def define_lr_scheduler(self, optimizer):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            if self.optim_dict["scheduler"] == 'cosine':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.optim_dict['num_epoch'], self.optim_dict['base_lr'] * 0.025)
            elif self.optim_dict["scheduler"] == 'multistep':
                lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.optim_dict['step'], gamma=self.optim_dict['gamma'])
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
