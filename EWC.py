import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EWC:
    def __init__(self, model, dataset_loader, importance=1000):
        self.model = model
        self.importance = importance
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataset_loader)
    
    def _compute_fisher(self, dataset_loader):
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)
        
        self.model.eval()
        for data, target in dataset_loader:
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            
            self.model.zero_grad()
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone().pow(2)
        
  
        for n in fisher:
            fisher[n] /= len(dataset_loader)
            
        return fisher
    
    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.importance * loss
