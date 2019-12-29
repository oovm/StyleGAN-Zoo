import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from wolframclient.serializers.serializable import WLSerializable

from sgan.net import *

LOADED_MODEL = {}


def get_model(name: str):
    m = name.lower()
    if m in LOADED_MODEL:
        return LOADED_MODEL[m]
    elif m == 'asuka':
        model = torch.hub.load('GalAster/StyleGAN-Zoo', 'style_asuka', pretrained=True)
        LOADED_MODEL[m] = model
        return model
    else:
        return 0


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
        return self.data

    def show(self):
        img = self.output()[0].permute(1, 2, 0)
        plt.imshow(img * 0.5 + 0.5)

    def save(self, path: str):
        pass

    def to_wl(self):
        return ToPILImage()(self.output()[0])

    @staticmethod
    def new(method, gene, data):
        return StyleGAN(method, gene=gene, data=data)


if __name__ == "__main__":
    s = StyleGAN('asuka')
    s.show()
