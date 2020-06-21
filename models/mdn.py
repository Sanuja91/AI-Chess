import torch, os, cv2
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions.distribution import Distribution
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from models.base import Neural_Network
from utilities import create_path

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
        self.seq_len = self.params['seq_len']
        self.learning_rate = self.params['learning_rate']
        self.grad_clip = self.params['grad_clip']
        self.batch_size = self.params['batch_size']
        self.device = self.get_device()
        
        self.lstm = nn.LSTM(self.z_size + self.action_size, self.hidden_size, self.stacked_layers, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, self.gaussian_size * (self.z_size + self.action_size))
        self.fc2 = nn.Linear(self.hidden_size, self.gaussian_size * (self.z_size + self.action_size))
        self.fc3 = nn.Linear(self.hidden_size, self.gaussian_size * (self.z_size + self.action_size))
        
        if load_model != False:
            self.load_state_dict(self.weights)
        
        print(self, "\n\n")

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.gaussian_size, self.z_size + self.action_size)
        mu = mu.view(-1, rollout_length, self.gaussian_size, self.z_size + self.action_size)
        sigma = sigma.view(-1, rollout_length, self.gaussian_size, self.z_size + self.action_size)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
        
        
    def forward(self, x, h):
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
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
    z = torch.load(create_path('data\\inputs\\tensors\\zs.pt')).float()
    actions = torch.load(create_path('data\\inputs\\tensors\\actions.pt')).float()
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


legal_actions = ['g1h3','g1f3','b1c3','b1a3','h2h3','g2g3','f2f3','e2e3','d2d3','c2c3','b2b3','a2a3','h2h4','g2g4','f2f4','e2e4','d2d4','c2c4','b2b4','a2a4']
'#  'Q NE1'
#  'Q NE2',
#  'Q NE3',
#  'Q NE4',
#  'Q NE5',
#  'Q NE6',
#  'Q NE7',
#  'Q E1',
#  'Q E2',
#  'Q E3',
#  'Q E4',
#  'Q E5',
#  'Q E6',
#  'Q E7',
#  'Q SE1',
#  'Q SE2',
#  'Q SE3',
#  'Q SE4',
#  'Q SE5',
#  'Q SE6',
#  'Q SE7',
#  'Q S1',
#  'Q S2',
#  'Q S3',
#  'Q S4',
#  'Q S5',
#  'Q S6',
#  'Q S7',
#  'Q SW1',
#  'Q SW2',
#  'Q SW3',
#  'Q SW4',
#  'Q SW5',
#  'Q SW6',
#  'Q SW7',
#  'Q W1',
#  'Q W2',
#  'Q W3',
#  'Q W4',
#  'Q W5',
#  'Q W6',
#  'Q W7',
#  'Q NW1',
#  'Q NW2',
#  'Q NW3',
#  'Q NW4',
#  'Q NW5',
#  'Q NW6',
#  'Q NW7',
#  'K N',
#  'K NE',
#  'K E',
#  'K SE',
#  'K S',
#  'K SW',
#  'K W',
#  'K NW'
# }