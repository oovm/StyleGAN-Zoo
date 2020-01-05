StyleGAN Zoo
============

Base on https://github.com/podgorskiy/StyleGAN_Blobless

## Install

Pytorch needed, install via conda first

```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
pip install stylegan_zoo
```


## Start

```python
import sgan
from sgan import StyleGAN


a = StyleGAN('asuka')
a.show()

b = sgan.generate('asuka', 5)
```

- For jupyter:

![](https://user-images.githubusercontent.com/17541209/71554236-b0813300-2a57-11ea-9ee4-fab29d592d9a.png)

- For mathematica:

![](https://user-images.githubusercontent.com/17541209/71553454-c5a39500-2a4a-11ea-8513-7d9a475c4c46.png)

- Multi-generation

![](https://user-images.githubusercontent.com/17541209/71593157-df89c880-2b6d-11ea-8455-8dd4d2024671.png)

- Style-interpolate

![](https://user-images.githubusercontent.com/17541209/71773895-45c48000-2fa0-11ea-8068-d7e5347a8233.png)


## License

| Part         | License                        |
| :----------- | :----------------------------- |
| Code         | [Apache License Version 2.0]() |
| [Asuka]()    | [CC0 - Creative Commons]()     |
| [Horo]()     | [CC0 - Creative Commons]()     |
| [Baby]()     | [CC4.0 Non-Commercial]()       |
| [FFHQ]()     |                                |
| [CelebaHQ]() |                                |
