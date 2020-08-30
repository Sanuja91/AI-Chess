import torch
import torch.nn as nn
# from cma import CMAEvolutionStrategy
from torch.multiprocessing import Process, Queue

from models.base import Neural_Network
from models.vae import Conv

class Policy_Controller(Neural_Network):
    """A Controller using Fully Connected Layers"""

    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """
        super(Policy_Controller, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'Policy Controller'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params

        self.z_size = self.params['z_size']
        self.hidden_size = self.params['hidden_size']
        self.action_size = self.params['action_size']
        self.device = self.get_device()
        
        self.fc1 = nn.Sequential(nn.Linear(self.z_size + self.hidden_size, self.params['expansion_size']), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.params['expansion_size'], 512), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())

        self.decoder = nn.Sequential(
            Conv(256, 256, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = True),
            Conv(256, 128, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = True),
            Conv(128, 64, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = True),
            Conv(64, self.action_size[0], 2, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = True),
#             Conv(32, 3, 6, 2, 0, conv = nn.ConvTranspose2d, activation = nn.Sigmoid, batch_norm = False)
        )
        
        if load_model != False:
            self.load_state_dict(self.weights)
        
        print(self, "\n\n")

    def forward(self, x):
        # print("ACTOR INPUT", x.shape)
        xp = self.fc1(x)
        xp = self.fc2(xp)
        xp = self.fc3(xp)
        # print("ACTOR FC1", xp.shape)
        xp = xp.reshape(xp.shape[0], -1, 1, 1)
        # print("ACTOR RESHAPE", xp.shape)
        xp = self.decoder(xp)
        # print("ACTOR DECODER", xp.shape)
        return xp


class Value_Controller(Neural_Network):
    """A Controller using Fully Connected Layers"""

    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """
        super(Value_Controller, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'Value Controller'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params

        self.z_size = self.params['z_size']
        self.hidden_size = self.params['hidden_size']
        self.action_size = self.params['action_size']
        self.device = self.get_device()
        
        self.fc1 = nn.Sequential(nn.Linear(self.z_size + self.hidden_size, 100), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(100, 20), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(20, 1))
        
        if load_model != False:
            self.load_state_dict(self.weights)
        
        print(self, "\n\n")

    def forward(self, x):    
        # print("CRITIC INPUT", x.shape)
        xv = self.fc1(x)
        # print("CRITIC FC1", xv.shape)
        xv = self.fc2(xv)
        # print("CRITIC FC2", xv.shape)
        xv = self.fc3(xv)
        # print("CRITIC FC3", xv.shape)
        return xv