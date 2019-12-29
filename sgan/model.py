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

import torch
from torch import nn
import random
from sgan.net import Generator, Mapping
import numpy as np


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class Model(nn.Module):
    def __init__(
            self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
            truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, model='normal'
    ):
        super(Model, self).__init__()
        self.model = model
        self.out_layer = layer_count - 1
        self.mapping = Mapping(
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers
        )

        self.generator = Generator(
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels
        )

        self.dlatent_avg = DLatent(latent_size, self.mapping.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

    def generate(self, lod, remove_blob=True, z=None, count=32):
        if z is None:
            z = torch.randn(count, self.latent_size)
        styles = self.mapping(z)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping(z2)

                layer_idx = torch.arange(self.mapping.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if self.truncation_psi is not None:
            layer_idx = torch.arange(self.mapping.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.generator.forward(styles, lod, remove_blob, method=self.model)
        return rec

    def forward(self, x, lod, blend_factor, d_train):
        return self.generate(x, lod, blend_factor, d_train)
