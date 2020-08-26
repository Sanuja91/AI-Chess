import torch, os, random
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
        self.epoch = 0
        self.train_steps = 0
        super(Neural_Network, self).__init__()
    
    def get_device(self):
        """Checks if GPU is available else runs on CPU"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\nAI Running on {}\n'.format(device))
        
        if str(device) == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        print('WARNING OVERRIDDING GPU TO RUN ON CPU')
        return 'cpu'

    def load_model(self, file_name):
        """Loads the models parameters and weights"""
        path = f'checkpoints_/{self.type}/{self.name}/'
        

        if file_name == 'Latest':
            list_of_file_paths = [path + x for x in os.listdir(path)]
            latest_file = max(list_of_file_paths, key = os.path.getctime)
            _, file_name = os.path.split(latest_file)

        elif file_name == 'Random':
            list_of_files = [x for x in os.listdir(path) if '.pth' in x]
            file_name = random.choice(list_of_files)
            print(f'Loading Random file {file_name}')

        path = f'checkpoints_/{self.type}/{self.name}/{file_name}'
        
        checkpoint = torch.load(path, map_location = lambda storage, loc: storage)
        self.params = checkpoint['params']
        self.weights = checkpoint['weights']
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'train_steps' in checkpoint:
            self.train_steps = checkpoint['train_steps']

    def save_model(self, epoch = 0, steps = 0):
        """Saves the models parameters and weights"""
        
        checkpoint = {
            'params' : self.params,
            'weights' : self.state_dict(),
            'epoch' : self.epoch,
            'train_steps' : self.train_steps
        }
        
        path = f'checkpoints_/{self.type}/{self.name}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        path += f'/E{epoch}.pth'
        torch.save(checkpoint, path)