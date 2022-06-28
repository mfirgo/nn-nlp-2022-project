from itertools import chain
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform
from torch.distributions import constraints
import torch.optim as optim
import torch.utils.data 

# Imports for plotting
import numpy as np
import matplotlib.pyplot as plt
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MainModel(nn.Module):
    def __init__(self, dim, blocks, flow_length, base_distributions, transition_matrix=None, device='cpu'):
        super().__init__()
        
        self.flows = []
        for d in base_distributions:
            self.flows.append(NormalizingFlow(dim=dim,device=device,blocks=blocks,flow_length=flow_length,
                                              base_distrib=d))    
        
        self.device = device
        self.hmm = HMM(len(self.flows),self.flows,transition_matrix=transition_matrix)
    
    def forward(self, x, T):
        return self.hmm(x, T)
    
    def parameters(self):
        return chain(*[f.parameters() for f in self.flows])
    
    def to(self, device):
        super().to(device)
        self.hmm = self.hmm.to(device)

        self.flows = [f.to(device) for f in self.flows]
            
        return self

def hmm_step(hmm_object, x, T):
    with torch.no_grad():
        hmm_object.forward_backward_step(x, T)


def nf_step(model, x, T, optimizer, scheduler):
    optimizer.zero_grad()
    
    loss = - torch.mean(model(x, T))
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle


def train_MainModel(model, dataset, batch_size, epochs, optimizer, scheduler):
    global copyed_model
    model.train()
    dataloader = DataLoader(dataset,batch_size=batch_size)
    
    for epoch in tqdm(range(epochs)):
        turn = 0
        
        for batch, x in enumerate(dataloader):
            loc_batch_size = x.shape[0]
            T = torch.ones([batch_size,1], dtype=torch.int64, device=device)*my_dataset.sequence_length
            loss = nf_step(model, x, T, optimizer, scheduler)
            turn += 1
            if turn%16==1:
                c = 0
                dataloader2 = DataLoader(dataset,batch_size=128,shuffle=True)
                for batch, x in enumerate(dataloader2):
                    T = torch.ones([128,1], dtype=torch.int64, device=device)*my_dataset.sequence_length
                    hmm_step(model.hmm, x, T)
                    c+=1
                    if c==2:
                        break
        with torch.no_grad():
            print(- torch.mean(model(x, T)))
        copyed_model = pickle.loads(pickle.dumps(main_model))
        y, states = copyed_model.hmm.sample(100)
                    
            
        copyed_model = pickle.loads(pickle.dumps(main_model))
        y, states = copyed_model.hmm.sample(100)
        plt.scatter(x[:,0].detach().cpu().numpy(), x[:,1].detach().cpu().numpy())
        plt.show()