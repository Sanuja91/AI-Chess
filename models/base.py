import torch, os
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from time import time

class Neural_Network(nn.Module):
    """Base layer for all neural network models"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        ======
        * **name** (string) --- name of the model
        * **type_** (string) --- type of the model
        * **params** (dict-like) --- a dictionary of parameters
        """
        super(Neural_Network, self).__init__()
    
    def get_device(self):
        """Checks if GPU is available else runs on CPU"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\nAI Running on {}\n'.format(device))
        
        if str(device) == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        return device

    def load_model(self, file_name):
        """Loads the models parameters and weights"""
        path = f'checkpoints/{self.type}/{self.name}/'
        print(path)

        if file_name == 'Latest':
            list_of_files = [path + x for x in os.listdir(path)]
            latest_file = max(list_of_files, key = os.path.getctime)
            _, file_name = os.path.split(latest_file)

        path = f'checkpoints/{self.type}/{self.name}/{file_name}'
        
        checkpoint = torch.load(path, map_location = lambda storage, loc: storage)
        self.params = checkpoint['params']
        self.weights = checkpoint['weights']

    def save_model(self):
        """Saves the models parameters and weights"""
        
        checkpoint = {
            'params' : self.params,
            'weights' : self.state_dict()
        }
        
        path = f'checkpoints/{self.type}/{self.name}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        path += f'/{str(time())}.pth'
        torch.save(checkpoint, path)