import torch
from sgan.model import Model as _m

dependencies = ['torch']


def style_asuka(pretrained=False):
    model = _m(
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


def style_horo(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.1.0/Horo-512x512.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def style_ffhq(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.2.0/FFHQ-1024x1024.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def style_celeba_hq(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.3.0/CelebaHQ-1024x1024.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def style_baby(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.8.0/Baby-1024x1024.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def style_wanghong(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.4.0/WangHong-1024x1024.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def style_asian_people(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.5.0/AsianPeople-1024x1024.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def style_asian_star(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.6.0/AsianStar-1024x1024.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def style_super_star(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        model='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.7.0/SuperStar-1024x1024.mat'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model
