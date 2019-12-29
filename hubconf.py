import torch
from sgan.model import Model as _m

dependencies = ['torch']


def style_asuka(pretrained=False):
    model = _m(
        channels=3,
        mapping_layers=8,
        latent_size=512,

        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.0.0/Asuka-512x512.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model
