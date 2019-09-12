# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sgan.lreq as ln
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)


def style_mod(x, style):
    style = style.view(style.shape[0], 2, x.shape[1], 1, 1)
    return torch.addcmul(style[:, 1], value=1.0, tensor1=x, tensor2=style[:, 0] + 1)


def upscale2d(x, factor=2):
    # s = x.shape
    # x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    # x = x.repeat(1, 1, 1, factor, 1, factor)
    # x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    # return x
    return F.upsample(x, scale_factor=factor, mode='bilinear', align_corners=True)


class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, latent_size, has_first_conv=True, fused_scale=True, layer=0):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        self.fused_scale = fused_scale
        if has_first_conv:
            if fused_scale:
                self.conv_1 = ln.ConvTranspose2d(inputs, outputs, 3, 2, 1, bias=False, transform_kernel=True)
            else:
                self.conv_1 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.blur = Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_1 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_1 = ln.Linear(latent_size, 2 * outputs, gain=1)

        self.conv_2 = ln.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.instance_norm_2 = nn.InstanceNorm2d(outputs, affine=False, eps=1e-8)
        self.style_2 = ln.Linear(latent_size, 2 * outputs, gain=1)

        self.layer = layer

        self.c = -1

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def set(self, c):
        self.c = c

    def forward(self, x, s1, s2):
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)

        x = torch.addcmul(
            x, value=1.0, tensor1=self.noise_weight_1,
            tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]])
        )

        x = x + self.bias_1

        x = F.leaky_relu(x, 0.2)

        x = self.instance_norm_1(x)

        x = style_mod(x, self.style_1(s1))

        x = self.conv_2(x)

        x = torch.addcmul(
            x, value=1.0, tensor1=self.noise_weight_2,
            tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]])
        )

        x = x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm_2(x)

        x = style_mod(x, self.style_2(s2))

        return x

    def forward_double(self, x, _x, s1, s2):
        if self.has_first_conv:
            if not self.fused_scale:
                x = upscale2d(x)
                _x = upscale2d(_x)
            x = self.conv_1(x)
            _x = self.conv_1(_x)

            x = self.blur(x)
            _x = self.blur(_x)

        n1 = torch.randn([int(x.shape[0]), 1, int(x.shape[2]), int(x.shape[3])])
        x = torch.addcmul(
            x, value=1.0, tensor1=self.noise_weight_1,
            tensor2=n1
        )

        _x = torch.addcmul(
            _x, value=1.0, tensor1=self.noise_weight_1,
            tensor2=n1
        )

        x = x + self.bias_1
        _x = _x + self.bias_1

        x = F.leaky_relu(x, 0.2)
        _x = F.leaky_relu(_x, 0.2)

        std = x.std(axis=[2, 3], keepdim=True)
        mean = x.mean(axis=[2, 3], keepdim=True)

        x = (x - mean) / std
        _x = (_x - mean) / std

        x = style_mod(x, self.style_1(s1))
        _x = style_mod(_x, self.style_1(s1))

        x = self.conv_2(x)
        _x = self.conv_2(_x)

        n2 = torch.randn([int(x.shape[0]), 1, int(x.shape[2]), int(x.shape[3])])

        x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                          tensor2=n2)

        _x = torch.addcmul(_x, value=1.0, tensor1=self.noise_weight_2,
                           tensor2=n2)

        x = x + self.bias_2
        _x = _x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        _x = F.leaky_relu(_x, 0.2)

        std = x.std(axis=[2, 3], keepdim=True)
        mean = x.mean(axis=[2, 3], keepdim=True)

        x = (x - mean) / std
        _x = (_x - mean) / std

        x = style_mod(x, self.style_2(s2))
        _x = style_mod(_x, self.style_2(s2))

        return x, _x


class ToRGB(nn.Module):
    def __init__(self, inputs, channels):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = ln.Conv2d(inputs, channels, 1, 1, 0, gain=1)

    def forward(self, x):
        x = self.to_rgb(x)
        return x


class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels
        self.latent_size = latent_size

        mul = 2 ** (self.layer_count - 1)

        inputs = min(self.maxf, startf * mul)
        self.const = Parameter(torch.Tensor(1, inputs, 4, 4))
        self.zeros = torch.zeros(1, 1, 1, 1)
        init.ones_(self.const)

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        to_rgb = nn.ModuleList()

        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0
            fused_scale = resolution * 2 >= 128

            block = DecodeBlock(inputs, outputs, latent_size, has_first_conv, fused_scale=fused_scale, layer=i)

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            to_rgb.append(ToRGB(outputs, channels))

            # print("decode_block%d %s styles in: %dl out resolution: %d" % ((i + 1), millify(count_parameters(block)), outputs, resolution))
            self.decode_block.append(block)
            inputs = outputs
            mul //= 2

        self.to_rgb = to_rgb

    def decode(self, styles, lod, remove_blob=True):
        x = self.const
        _x = None
        for i in range(lod + 1):
            if i < 4 or not remove_blob:
                x = self.decode_block[i].forward(x, styles[:, 2 * i + 0], styles[:, 2 * i + 1])
                if remove_blob and i == 3:
                    _x = x.clone()
                    _x[x > 300.0] = 0

                # plt.hist((torch.max(torch.max(_x, dim=2)[0], dim=2)[0]).cpu().flatten().numpy(), bins=300)
                # plt.show()
                # exit()
            else:
                x, _x = self.decode_block[i].forward_double(x, _x, styles[:, 2 * i + 0], styles[:, 2 * i + 1])

        if _x is not None:
            x = _x
        if lod == 8:
            x = self.to_rgb[lod](x)
        else:
            x = x.max(dim=1, keepdim=True)[0]
            x = x - x.min()
            x = x / x.max()
            x = torch.pow(x, 1.0 / 2.2)
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, styles, lod, remove_blob=True):
        return self.decode(styles, lod, remove_blob)


class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul):
        super(MappingBlock, self).__init__()
        self.fc = ln.Linear(inputs, output, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


class Mapping(nn.Module):
    def __init__(self, num_layers, mapping_layers=5, latent_size=256, dlatent_size=256, mapping_fmaps=256):
        super(Mapping, self).__init__()
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.num_layers = num_layers
        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.01)
            inputs = outputs
            setattr(self, "block_%d" % (i + 1), block)

    def forward(self, z):
        x = pixel_norm(z)

        for i in range(self.mapping_layers):
            x = getattr(self, "block_%d" % (i + 1))(x)

        return x.view(x.shape[0], 1, x.shape[1]).repeat(1, self.num_layers, 1)
