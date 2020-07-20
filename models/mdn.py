import torch, os, cv2
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions.distribution import Distribution
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from models.base import Neural_Network

class MDN_RNN(Neural_Network):
    """A Mixed Density Network using an LSTM Core"""

    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """

        super(MDN_RNN, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'MDN RNN'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params
        self.z_size = self.params['z_size']
        self.action_size = self.params['action_size']
        self.hidden_size = self.params['hidden_size']
        self.gaussian_size = self.params['gaussian_size']
        self.stacked_layers = self.params['stacked_layers']
        self.learning_rate = self.params['learning_rate']
        self.grad_clip = self.params['grad_clip']
        self.batch_size = self.params['batch_size']
        self.device = self.get_device()
        
        self.lstm = nn.LSTM(self.z_size + self.action_size, self.hidden_size, self.stacked_layers, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, self.gaussian_size * self.z_size)
        self.fc2 = nn.Linear(self.hidden_size, self.gaussian_size * self.z_size)
        self.fc3 = nn.Linear(self.hidden_size, self.gaussian_size * self.z_size)
        
        if load_model != False:
            self.load_state_dict(self.weights)
        
        print(self, "\n\n")

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.gaussian_size, self.z_size)
        mu = mu.view(-1, rollout_length, self.gaussian_size, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.gaussian_size, self.z_size)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
        
        
    def forward(self, x, h, x_lengths = None):
        if x_lengths is not None:
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first = True, enforce_sorted = True)
        
        y, (h, c) = self.lstm(x, h)
        
        if x_lengths is not None:
            # undo the packing operation
            y, _ = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first = True)

        pi, mu, sigma = self.get_mixture_coef(y)
        rsoureturn (pi, mu, sigma), (h, c)
    
    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.stacked_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.stacked_layers, batch_size, self.hidden_size).to(self.device)
        )


def loss_function(y, pi, mu, sigma):
    y = y.unsqueeze(2)
    m = torch.distributions.Normal(loc = mu, scale = sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim = 2)
    loss = -torch.log(loss)
    return loss.mean()

def train_mdn(mdn, epochs, log_interval):
    """Trains the MDN RNN"""

    mdn.train()
    mdn = mdn.to(mdn.device)
    optimizer = optim.Adam(mdn.parameters(), lr = mdn.learning_rate)
    z = torch.load('data/inputs/tensors/zs.pt').float()
    actions = torch.load('data/inputs/tensors/actions.pt').float()
    z = torch.cat((z, actions), dim = 1).unsqueeze(0)    
    
    # Creates batches
    inputs = []
    targets = []
    for i in range(0, z.size(1) - mdn.seq_len, mdn.seq_len):
        # Get mini-batch inputs and targets
        inputs.append(z[ : , i : i + mdn.seq_len, : ])
        targets.append(z[ : , (i + 1) : (i + 1) + mdn.seq_len, : ])

    if mdn.batch_size > len(inputs): 
        mdn.batch_size = len(inputs)
    else:
        raise Exception('Code for handling larger batch sizes not written.. Easy fix')
    
    inputs = torch.stack(inputs).squeeze(1).to(mdn.device)
    targets = torch.stack(targets).squeeze(1).to(mdn.device)

    for epoch in range(epochs):
        # Set initial hidden and cell states
        hidden = mdn.init_hidden(mdn.batch_size)

        # Forward pass
        hidden = (hidden[0].detach(), hidden[1].detach())
        (pi, mu, sigma), hidden = mdn(inputs, hidden)
        loss = loss_function(targets, pi, mu, sigma)

        # Backward and optimize
        mdn.zero_grad()
        loss.backward()
        clip_grad_norm_(mdn.parameters(), mdn.grad_clip)
        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, epochs, loss.item()))
    mdn.save_model()