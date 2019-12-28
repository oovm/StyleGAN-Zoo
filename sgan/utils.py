import torch
from sgan.net import *
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt


class StyleGAN:
    def __init__(self, method: str, gene=None, data=None):
        self.method = method
        self.data = data
        self.gene = gene

    def output(self, device='cpu'):
        if self.gene is None:
            latents = torch.randn(1, 512)
            self.gene = torch.tensor(latents).float().to(device)
        if self.data is None:
            with torch.no_grad():
                model = get_model(self.method)
                model.to(device)
                self.data = model.generate(7, z=self.gene)
        return self.data

    def show(self):
        ToPILImage()(self.output())

    def save(self, path: str):
        pass

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
