import os
import re
import math
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from wolframclient.serializers.serializable import WLSerializable
from torchvision.utils import save_image
from numpy import savetxt
from sgan.net import *

LOADED_MODEL = {}
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


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
        self.gene = gene

    def output(self, device='cpu'):
        if self.gene is None:
            latents = torch.randn(1, 512)
            self.gene = torch.tensor(latents).float().to(device)
        if self.data is None:
            with torch.no_grad():
                model = get_model(self.method)
                model.to(device)
                self.data = model.generate(model.out_layer, z=self.gene)
            self.gene = self.gene.cpu()
            self.data = self.data.cpu()
        return self.data

    def forward(self, device='cpu', truncation_psi=0.5):
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

    def to_wl(self):
        img = self.output()[0].clamp(-1, 1)
        return ToPILImage()(img * 0.5 + 0.5)

    @staticmethod
    def new(method, gene, data):
        return StyleGAN(method, gene=gene, data=data)


def generate(
        method, num,
        device='cuda', save=None,
        batch_size=16,
        truncation_psi=0.75
):
    # prepare model
    model = get_model(method)
    model.truncation_psi = truncation_psi
    model.to(device)
    # batch eval
    with torch.no_grad():
        gene = []
        data = []
        for i in range(math.ceil(num / batch_size)):
            latents = torch.randn(batch_size, 512).to(device)
            batch = model.generate(model.out_layer, z=latents)
            if save is not None:
                o = [StyleGAN.new(method, i.unsqueeze(0), j.unsqueeze(0)) for i, j in zip(latents.cpu(), batch.cpu())]
                for j in o:
                    j.save(path=save)
            else:
                gene.append(latents.cpu())
                data.append(batch.cpu())
    if save is None:
        gene = torch.stack(gene, dim=1)
        data = torch.stack(data, dim=1)
        o = [StyleGAN.new(method, i.unsqueeze(0), j.unsqueeze(0)) for i, j in zip(gene, data)]
        return o[:num]
    else:
        pass


def reinitialize():
    global LOADED_MODEL
    LOADED_MODEL = {}
    torch.hub.list('GalAster/StyleGAN-Zoo', force_reload=True)


if __name__ == "__main__":
    # s = StyleGAN('asuka')
    # s.output(device='cuda')
    # s.show()
    # s.save('.')
    m = generate('asuka', 5)
    # generate('asuka', 2, save='.')
