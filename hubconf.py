from torch.hub import load_state_dict_from_url as _download
from sgan.model import Model as _m

dependencies = ['torch']


def style_asuka(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.0.0/Asuka-512x512-7de0e6.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_horo(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.1.0/Horo-512x512-822ee4.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_asashio(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.2.0/Asashio-512x512-a3c21a.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_anime_head(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.1.0/AnimeHead-512x512-960a82.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_anime_face_a(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.0.0/AnimeFaceC-512x512-47055c.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_anime_face_b(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.0.0/AnimeFaceA-512x512-feaff1.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_anime_face_c(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.0.0/AnimeFaceB-512x512-41bdee.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_anime_face_d(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.5,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.0.0/AnimeFaceD-512x512-3e59ff.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_anime_face_e(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.4,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.0.0/AnimeFaceE-512x512-9cfc38.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_art_a(pretrained=False):
    model = _m(
        layer_count=8,
        startf=32,
        maxf=512,

        truncation_psi=0.4,
        truncation_cutoff=8,
        mode='asuka'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.0.0/AnimeFaceE-512x512.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_ffhq(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.2.0/FFHQ-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_celeba_hq(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.3.0/CelebaHQ-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_baby(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.8.0/Baby-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_wanghong(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.4.0/WangHong-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_asian_people(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.5.0/AsianPeople-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_asian_star(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.6.0/AsianStar-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_super_star(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v1.7.0/SuperStar-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_vessel(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.3.0/Vessel-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model


def style_qinghua(pretrained=False):
    model = _m(
        layer_count=9,
        startf=16,
        maxf=512,

        truncation_psi=0.75,
        truncation_cutoff=8,
        mode='normal'
    )
    if pretrained:
        checkpoint = 'https://github.com/GalAster/StyleGAN-Zoo/releases/download/v2.3.1/QingHua-1024x1024.mat'
        model.load_state_dict(_download(checkpoint, progress=True))
    return model
