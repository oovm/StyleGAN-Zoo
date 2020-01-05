import os
import errno
import warnings
import torch
import sys
import re

from torch.hub import load as loading
from torch.hub import _download_url_to_file, _get_torch_home, urlparse

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
    elif m == 'animehead':
        model = loading('GalAster/StyleGAN-Zoo', 'style_anime_head', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'animefacea':
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
    elif m == 'asianstar':
        model = loading('GalAster/StyleGAN-Zoo', 'style_asian_star', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'superstar':
        model = loading('GalAster/StyleGAN-Zoo', 'style_super_star', pretrained=True)
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


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True):
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        _download_url_to_file(url, cached_file, '000000', progress=progress)
    return torch.load(cached_file, map_location=map_location)


def reinitialize(model=None):
    global LOADED_MODEL
    LOADED_MODEL = {}
    torch.hub.list('GalAster/StyleGAN-Zoo', force_reload=True)
    if model is not None:
        # remove model
        pass
