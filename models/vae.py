import torch, os, cv2, pickle
from copy import copy
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions.distribution import Distribution
import matplotlib.pyplot as plt
from models.base import Neural_Network

class Conv(nn.Module):
    """Simlifed NN for ease of code"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv = nn.Conv1d, activation = nn.ReLU, batch_norm = False):
        """Initialize an Convolutional model given a dictionary of parameter.s

        Params
        =====
        * **in_channels** (int) --- input channels
        * **out_channels** (int) --- output channels
        * **kernel_size** (int) --- feature map size
        * **stride** (int) --- stride length
        * **padding** (int) --- padding
        * **conv** (function) --- convulational function that is to be used
        * **activation** (function) --- activation function that is to be used
        * **batch_norm** (boolean) --- whether batch normalization is required
        """

        super(Conv, self).__init__()
        self.conv = conv(
            in_channels, 
            out_channels,
            kernel_size = kernel_size, 
            stride = stride,
            padding = padding, 
            bias = False
        ) 
        
        if batch_norm:
            if conv == nn.Conv2d:
                self.bn = nn.BatchNorm2d(
                    out_channels,
                    eps = 0.001, # value found in tensorflow
                    momentum = 0.1, # default pytorch value
                    affine = True
                )
            else:
                self.bn = nn.BatchNorm1d(
                    out_channels,
                    eps = 0.001, # value found in tensorflow
                    momentum = 0.1, # default pytorch value
                    affine = True
                )
        else:
            self.bn = None

        self.activation = activation()

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x



class CNN_VAE(Neural_Network):
    """Convolutional Variational Auto Encoder"""

    def __init__(self, name, params, load_model = False):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """
        
        super(CNN_VAE, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'Conv VAE'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params
        self.z_size = self.params['z_size']
        self.batch_size = self.params['batch_size']
        self.learning_rate = self.params['learning_rate']
        self.kl_tolerance = self.params['kl_tolerance']
        self.batch_norm = self.params['batch_norm']
        self.starting_channels = self.params['starting_channels']
        self.device = self.get_device()
        
        # The Encoder
        self.encoder = nn.Sequential(
            Conv(self.starting_channels, 32, 1, 1, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = self.batch_norm),
            Conv(32, 64, 3, 1, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = self.batch_norm),
            Conv(64, 128, 3, 1, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = self.batch_norm),
            Conv(128, 256, 3, 1, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = self.batch_norm),
            Conv(256, 512, 2, 1, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = self.batch_norm)
        )

        self.enc_conv_mu = Conv(512, self.z_size, 1, 1, 0, conv = nn.Conv1d, activation = nn.ReLU, batch_norm = self.batch_norm)
        self.enc_conv_logvar = Conv(512, self.z_size, 1, 1, 0, conv = nn.Conv1d, activation = nn.ReLU, batch_norm = self.batch_norm)

        # channels = [self.starting_channels, 16, 32]
        # reverse_channels = channels[::-1]

        # self.encoder = nn.Sequential(
        #     Conv(channels[0], channels[1], 3, 2, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = self.batch_norm),
        #     Conv(channels[1], channels[2], 3, 2, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = self.batch_norm),
        # )
    
        # self.enc_conv_mu = Conv(channels[-1], self.z_size, 1, 1, 0, conv = nn.Conv1d, activation = nn.ReLU, batch_norm = self.batch_norm)
        # self.enc_conv_logvar = Conv(channels[-1], self.z_size, 1, 1, 0, conv = nn.Conv1d, activation = nn.ReLU, batch_norm = self.batch_norm)

        self.epsilon = Distribution((self.batch_size, self.z_size))

        # The Decoder
        self.decoder = nn.Sequential(
            Conv(512, 256, 2, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(256, 128, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(128, 64, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(64, 32, 3, 1, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(32, self.starting_channels, 1, 1, 0, conv = nn.ConvTranspose2d, activation = nn.Sigmoid, batch_norm = False)
        )

        # self.decoder = nn.Sequential(
        #     Conv(reverse_channels[0], reverse_channels[1], 3, 2, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
        #     Conv(reverse_channels[1], reverse_channels[2], 4, 2, 1, conv = nn.ConvTranspose2d, activation = nn.Sigmoid, batch_norm = False)
        # )

        # self.dec_conv = Conv(self.z_size, reverse_channels[0], 1, 1, 0, conv = nn.Conv1d, activation = nn.ReLU, batch_norm = self.batch_norm)
      
        self.dec_conv = Conv(self.z_size, 512, 1, 1, 0, conv = nn.Conv1d, activation = nn.ReLU, batch_norm = self.batch_norm)
        
        if load_model != False:
            self.load_state_dict(self.weights)
        print(self, "\n\n")


    def forward(self, x):
        # print('INPUT', x.shape)
        mu, logvar = self.encode(x)
        # print('MU', mu.shape, "LOG VAR", logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print("Z SHAPE", z.shape)
        x = self.decode(z)
        # print("DECODER", x.shape)
        return x, mu, logvar

    def encode(self, x):
        """Encodes observation"""
        x = self.encoder(x)
        # print('ENCODER', x.shape)
        x.squeeze_(-1)
        # print('ENCODER RESHAPE', x.shape)
        mu = self.enc_conv_mu(x)
        logvar = self.enc_conv_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterizes the tensors"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodes the encoding"""
        x = self.dec_conv(z)
        # print('DEC CONV', x.shape)
        x.unsqueeze_(-1)
        # print('DECODER RESHAPE', x.shape)
        return self.decoder(x)


def loss_function(outputs, inputs, mu, logvar):
    """Reconstruction + KL divergence losses summed over all elements and batch"""
    # BCE = F.binary_cross_entropy(outputs, inputs.view(-1, 2 * 2 * 256), reduction = 'sum')
    BCE = F.binary_cross_entropy(outputs.view(-1), inputs.view(-1), reduction = 'sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(vae, data, epochs, log_interval):
    """Trains the Conv VAE"""
    vae.train()
    vae = vae.to(vae.device)
    optimizer = optim.Adam(vae.parameters(), lr = vae.learning_rate)
    train_loss = 0
    train_loader = DataLoader(
        torch.tensor(data).float(),
        batch_size = vae.batch_size, 
        shuffle = True
    )
    count = 0
    for epoch in range(epochs):
        # for batch_idx, (inputs, _) in enumerate(train_loader):
        for batch_idx, inputs in enumerate(train_loader):
            if inputs.shape[0] != vae.batch_size:
                continue
            inputs = inputs.to(vae.device)
            optimizer.zero_grad()
            outputs, mu, logvar = vae(inputs)
            # print(outputs.shape, inputs.shape, mu.shape, logvar.shape)
            loss = loss_function(outputs, inputs, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            count += 1
            if count % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(inputs)))
                path = f'data/outputs/{vae.type}/{vae.name}'
                if not os.path.exists(path):
                    os.makedirs(path)
                sample_input = inputs[0].squeeze(0).detach().cpu()
                sample_output = outputs[0].squeeze(0).detach().cpu()
                
                sample_input = transforms.ToPILImage(mode = 'RGB')(sample_input)
                sample_input.save(f'{path}/E - {epoch} B - {batch_idx} target.png',"PNG")
                sample_output = transforms.ToPILImage(mode = 'RGB')(sample_output)
                sample_output.save(f'{path}/E - {epoch} B - {batch_idx} output.png',"PNG")
                
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    vae.save_model()