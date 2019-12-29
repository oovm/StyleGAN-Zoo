StyleGAN Zoo
============

Base on https://github.com/podgorskiy/StyleGAN_Blobless

## Install

Pytorch needed, install via conda first

```shell
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install stylegan_zoo
```


## Start

```python
from sgan import StyleGAN

s = StyleGAN('asuka')
s.show()
```

For jupyter:

![](https://user-images.githubusercontent.com/17541209/71553336-29c55980-2a49-11ea-8e94-18b7ab384706.png)

For mathematica:

![](https://user-images.githubusercontent.com/17541209/71553335-29c55980-2a49-11ea-83bf-56b3fd7bb35f.png)

