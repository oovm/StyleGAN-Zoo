import os
import re
import math
import matplotlib.pyplot as plt
import torch

from torchvision.transforms import ToPILImage
from wolframclient.serializers.serializable import WLSerializable
from torchvision.utils import save_image
from numpy import savetxt

LOADED_MODEL = {}
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    DEFAULT_DEVICE = 'cuda'
else:
    DEFAULT_DEVICE = 'cpu'


def get_model(name: str):
    m = re.sub('[-_ ]', '', name).lower()
    if m in LOADED_MODEL:
        return LOADED_MODEL[m]
    elif m == 'asuka':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_asuka', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'horo':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_horo', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'baby':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_baby', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'wanghong':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_wanghong', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'asianpeople':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_asian_people', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'asianstar':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_asian_star', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'superstar':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_super_star', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    elif m == 'ffhq':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_ffhq', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    else:
        return AttributeError()


class StyleGAN(WLSerializable):
    def __init__(self, method: str, gene=None, data=None):
        self.method = method
        self.data = data
        if gene is None:
            latents = torch.randn(1, 512)
            self.gene = torch.tensor(latents).float()
        else:
            self.gene = gene

    def output(self):
        if self.data is None:
            with torch.no_grad():
                model = get_model(self.method)
                model.to(DEFAULT_DEVICE)
                self.data = model.generate(model.out_layer, z=self.gene)
            self.gene = self.gene.cpu()
            self.data = self.data.cpu()
        return self.data

    '''
    def forward(self, device=DEFAULT_DEVICE, truncation_psi=0.5):
        """
        This will permanently change the network settings!
        """
        if self.gene is None:
            self.gene = torch.randn(1, 512).to(device)
        with torch.no_grad():
            model = get_model(self.method)
            model.truncation_psi = truncation_psi
            model.to(device)
            self.data = model.generate(model.out_layer, z=self.gene)
            self.gene = self.gene.cpu()
            self.data = self.data.cpu()
        return self.data
    '''

    def show(self):
        img = self.output()[0].permute(1, 2, 0)
        plt.imshow(img.clamp(-1, 1) * 0.5 + 0.5)

    def save(self, path=None):
        img = self.output()[0] * 0.5 + 0.5
        src = self.gene
        name = str(src.__hash__())
        d = '.' if path is None else path
        savetxt(os.path.join(d, name + '-gene.txt'), src, delimiter='\n')
        save_image(img, os.path.join(d, name + '-show.png'))

    def clean(self):
        self.data = None

    def to_wl(self):
        img = self.output()[0].clamp(-1, 1)
        return ToPILImage()(img * 0.5 + 0.5)

    @staticmethod
    def new(method, gene, data):
        return StyleGAN(method, gene=gene, data=data)


def generate(
        method, num,
        save=None,
        batch_size=16
):
    # prepare model
    model = get_model(method)
    model.to(DEFAULT_DEVICE)
    # batch eval
    with torch.no_grad():
        gene = []
        data = []
        for i in range(math.ceil(num / batch_size)):
            latents = torch.randn(batch_size, 512).to(DEFAULT_DEVICE)
            batch = model.generate(model.out_layer, z=latents)
            if save is None:
                gene.append(latents.cpu())
                data.append(batch.cpu())
            else:
                o = [StyleGAN.new(method, i.unsqueeze(0), j.unsqueeze(0)) for i, j in zip(latents.cpu(), batch.cpu())]
                for j in o:
                    j.save(path=save)
    if save is None:
        gene = torch.cat(gene, dim=0)
        data = torch.cat(data, dim=0)
        o = [StyleGAN.new(method, i.unsqueeze(0), j.unsqueeze(0)) for i, j in zip(gene, data)]
        return o[:num]
    else:
        pass


def as_tensor(o):
    if isinstance(o, StyleGAN):
        o.output()
        return o.gene
    else:
        return o


def style_mix(model, genes, weights):
    pass


def slerp(start, end, values):
    low_norm = start / torch.norm(start, dim=1, keepdim=True)
    high_norm = end / torch.norm(end, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)

    def interpolate(val):
        s = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * start
        e = (torch.sin(val * omega) / so).unsqueeze(1) * end
        return s + e

    return list(map(interpolate, values))


def style_interpolate(
        a, b,
        method=None,
        steps=24,
        batch_size=16,
        save=None,
):
    i = slerp(as_tensor(a), as_tensor(b), [0.0, 0.25, 0.5, 0.75, 1])
    i = torch.cat(i, dim=0).to(DEFAULT_DEVICE)
    if method is None and isinstance(a, StyleGAN):
        method = a.method
    elif method is None and isinstance(b, StyleGAN):
        method = b.method
    model = get_model(method)
    model.to(DEFAULT_DEVICE)
    with torch.no_grad():
        result = model.generate(model.out_layer, z=i)
    o = [StyleGAN.new(method, i.unsqueeze(0), j.unsqueeze(0)) for i, j in zip(i.cpu(), result.cpu())]
    return o


def model_settings():
    pass


def image_encode():
    pass


def reinitialize(device='cuda'):
    global LOADED_MODEL
    global DEFAULT_DEVICE

    LOADED_MODEL = {}
    torch.hub.list('GalAster/StyleGAN-Zoo', force_reload=True)
    if device is 'cuda' and torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        DEFAULT_DEVICE = 'cuda'
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        DEFAULT_DEVICE = 'cpu'


if __name__ == "__main__":
    # test for normal
    '''
    t1 = StyleGAN('asuka')
    t1.output(device='cuda')
    t1.show()
    t1.save('.')
    '''
    # test for generate
    '''
    t2 = generate('asuka', 5, batch_size=2)
    t3 = generate('asuka', 2, save='.')
    '''
    # test for interpolate

    t4, t5 = generate('asuka', 2)
    out = style_interpolate(t4, t5, steps=4)
