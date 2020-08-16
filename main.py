import pickle
import numpy as np

from models.vae import CNN_AE, CNN_VAE, train_vae, train_ae

def train_vae_(name):
    """Trains the CNN-VAE"""
    params = {
        'z_size' : 400,
        'batch_size' : 10,
        'learning_rate' : 1e-4,
        'kl_tolerance' : 0.5,
        'batch_norm' : True,
        'starting_channels': 12
    }

    vae = CNN_AE(name, params, False)
    with open('data/vae/test_states.pkl',  'rb') as pickle_file:
        data = list(pickle.load(pickle_file).values())
        data = [np.array(data_) for data_ in data]
    
    train_ae(vae, data, 500, 100)




# train_vae_('Test')

def create_mdn_training_data(name):
    """Create training data for MDN-RNN"""
    vae = CNN_AE(name, None, 'Latest')
    vae.eval()
    train_loader = DataLoader(
        datasets.ImageFolder(
            'data\\inputs', 
            transform = transforms.ToTensor()),
        batch_size = vae.batch_size, 
        shuffle = False
    )

    path = f'data\\inputs\\tensors'
    if not os.path.exists(path):
        os.makedirs(path)

    mus = []
    logvars = []
    zs = []
    for batch_idx, (inputs, _) in enumerate(train_loader):
        mu, logvar = vae.encode(inputs)
        z = vae.reparameterize(mu, logvar)
        mus.append(mu)
        logvars.append(logvar)
        zs.append(z)

    mus = torch.cat(mus)
    logvars = torch.cat(logvars)
    zs = torch.cat(zs)

    torch.save(mus, f'{path}\\mus.pt')
    torch.save(logvars, f'{path}\\logvars.pt')
    torch.save(zs.squeeze(-1), f'{path}\\zs.pt')

train_vae_('Test')

