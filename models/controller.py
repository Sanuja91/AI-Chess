import torch
import torch.nn as nn
# from cma import CMAEvolutionStrategy
from torch.multiprocessing import Process, Queue

from models.base import Neural_Network
from models.vae import Conv

class Controller(Neural_Network):
    """A Controller using Fully Connected Layers"""

    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """
        super(Controller, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'Controller'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params
        self.z_size = self.params['z_size']
        self.hidden_size = self.params['hidden_size']
        self.action_size = self.params['action_size']
        self.device = self.get_device()
        
        self.fc1 = nn.Linear(self.z_size + self.hidden_size, self.params['expansion_size'])
        self.fc_v1 = nn.Linear(self.z_size + self.hidden_size, 100)
        self.fc_v2 = nn.Linear(100, 20)
        self.fc_v3 = nn.Linear(20, 1)
        
        self.decoder = nn.Sequential(
            Conv(self.params['expansion_size'], 256, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(256, 128, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(128, 64, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(64, self.action_size[0], 2, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
#             Conv(32, 3, 6, 2, 0, conv = nn.ConvTranspose2d, activation = nn.Sigmoid, batch_norm = False)
        )
        
        if load_model != False:
            self.load_state_dict(self.weights)
        
        print(self, "\n\n")

    def forward(self, x):
        xp = self.fc1(x)
#         print("XP FC1", xp.shape)
        xp = xp.reshape(xp.shape[0], -1, 1, 1)
#         print("XP RESHAPE", xp.shape)
        xp = self.decoder(xp)
#         print("XP DECODER", xp.shape)
    
        xv = self.fc_v1(x)
#         print("XV FC1", xv.shape)
        xv = self.fc_v2(xv)
#         print("XV FC2", xv.shape)
        xv = self.fc_v3(xv)
#         print("XV FC3", xv.shape)
        return xp, xv