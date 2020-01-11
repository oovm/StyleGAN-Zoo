import torch
import re

from torch.hub import load as loading

LOADED_MODEL = {}


def get_model(name: str):
    m = re.sub('[-_ ]', '', name).lower()
    if m in LOADED_MODEL:
        return LOADED_MODEL[m]
    elif m == 'asuka':
        model = loading('GalAster/StyleGAN-Zoo', 'style_asuka', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'horo':
        model = loading('GalAster/StyleGAN-Zoo', 'style_horo', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'asashio':
        model = loading('GalAster/StyleGAN-Zoo', 'style_asashio', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m in ['anime', 'animehead']:
        model = loading('GalAster/StyleGAN-Zoo', 'style_anime_head', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m in ['animeface', 'animefacea']:
        model = loading('GalAster/StyleGAN-Zoo', 'style_anime_face_a', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'animefaceb':
        model = loading('GalAster/StyleGAN-Zoo', 'style_anime_face_b', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'animefacec':
        model = loading('GalAster/StyleGAN-Zoo', 'style_anime_face_c', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'animefaced':
        model = loading('GalAster/StyleGAN-Zoo', 'style_anime_face_d', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'animefacee':
        model = loading('GalAster/StyleGAN-Zoo', 'style_anime_face_e', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'baby':
        model = loading('GalAster/StyleGAN-Zoo', 'style_baby', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'wanghong':
        model = loading('GalAster/StyleGAN-Zoo', 'style_wanghong', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'asianpeople':
        model = loading('GalAster/StyleGAN-Zoo', 'style_asian_people', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m in ['asian', 'asianstar']:
        model = loading('GalAster/StyleGAN-Zoo', 'style_asian_star', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m in ['star', 'superstar']:
        model = loading('GalAster/StyleGAN-Zoo', 'style_super_star', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m in ['art', 'arta']:
        model = loading('GalAster/StyleGAN-Zoo', 'style_art_a', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'artb':
        model = loading('GalAster/StyleGAN-Zoo', 'style_art_b', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m in ['artc', 'ukiyoefaces']:
        model = loading('GalAster/StyleGAN-Zoo', 'style_ukiyoe_faces', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'ffhq':
        model = loading('GalAster/StyleGAN-Zoo', 'style_ffhq', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'celebahq':
        model = loading('GalAster/StyleGAN-Zoo', 'style_celeba_hq', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'vessel':
        model = loading('GalAster/StyleGAN-Zoo', 'style_vessel', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'qinghua':
        model = loading('GalAster/StyleGAN-Zoo', 'style_qinghua', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    else:
        raise AttributeError()


def reinitialize(model=None):
    global LOADED_MODEL
    LOADED_MODEL = {}
    torch.hub.list('GalAster/StyleGAN-Zoo', force_reload=True)
    if model is not None:
        # remove model
        pass
