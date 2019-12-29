StyleGAN Zoo
============

Base on https://github.com/podgorskiy/StyleGAN_Blobless

## Install

Pytorch needed, install via conda first

```shell script
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install stylegan_zoo
```


## Start

```python
from sgan import StyleGAN

s = StyleGAN('asuka')
s.show()
```