from model import Model
from net import *
import torch
import numpy as np

cfg = {
    'DLATENT_AVG_BETA': 0.995,
    'LATENT_SPACE_SIZE': 512,
    'LAYER_COUNT': 8,
    'MAPPING_LAYERS': 8,
    'MAX_CHANNEL_COUNT': 512,
    'START_CHANNEL_COUNT': 32,
    'STYLE_MIXING_PROB': 0.9,
    'TRUNCATIOM_CUTOFF': 8,
    'TRUNCATIOM_PSI': 0.7,
}

model = Model(
    startf=cfg['START_CHANNEL_COUNT'],
    layer_count=cfg['LAYER_COUNT'],
    maxf=cfg['MAX_CHANNEL_COUNT'],
    latent_size=cfg['LATENT_SPACE_SIZE'],
    truncation_psi=cfg['TRUNCATIOM_PSI'],
    truncation_cutoff=cfg['TRUNCATIOM_CUTOFF'],
    mapping_layers=cfg['MAPPING_LAYERS'],
    channels=3
)

'''
data = torch.load('../Asuka-512x512.pth')['models']

model.generator.load_state_dict(data['generator_s'], strict=False)
model.mapping.load_state_dict(data['mapping_fl_s'], strict=False)
model.dlatent_avg.load_state_dict(data['dlatent_avg'], strict=False)

torch.save(model.state_dict(), '../Asuka-512x512.mat')
'''

rnd = np.random.RandomState(5)
model.load_state_dict(torch.load('../Asuka-512x512.mat'), strict=False)


class StyleGAN:
    def __init__(self, method: str, gene=None, data=None):
        self.method = method
        self.data = data
        self.gene = gene

    def output(self, device='cpu'):
        if self.gene is None:
            latents = rnd.randn(1, cfg['LATENT_SPACE_SIZE'])
            self.gene = torch.tensor(latents).float().to(device)
        if self.data is None:
            with torch.no_grad():
                self.data = model.to(device).generate(7, z=self.gene)
        return self.data

    @staticmethod
    def new(method, gene, data):
        return StyleGAN(method, gene=gene, data=data)


'''
with torch.no_grad():
    x_rec = model.generate(7, z=sample)
    resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
    resultsample = resultsample.cpu()[0, :, :, :]

    answer = resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)
'''
