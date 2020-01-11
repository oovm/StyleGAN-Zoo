StyleGAN Zoo
============
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/drive/1HHfyYCfnat4jhOnu34gqotRqzBiDeE_-)

Base on https://github.com/podgorskiy/StyleGAN_Blobless

Find models on https://github.com/GalAster/StyleGAN-Zoo/releases

## Install

Pytorch needed, install via conda first

```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
pip install stylegan_zoo
```


## Start

- For jupyter:

```python
from sgan import StyleGAN
a = StyleGAN('asuka')
a.show()
```

![](https://user-images.githubusercontent.com/17541209/71554236-b0813300-2a57-11ea-9ee4-fab29d592d9a.png)

- For mathematica:

```python
from sgan import StyleGAN

StyleGAN('asuka')
```

![](https://user-images.githubusercontent.com/17541209/71553454-c5a39500-2a4a-11ea-8513-7d9a475c4c46.png)

- Multi-generation

```python
from sgan import generate

generate('asuka', 4)
```

![](https://user-images.githubusercontent.com/17541209/71593157-df89c880-2b6d-11ea-8455-8dd4d2024671.png)

- Style-interpolate

```python
from sgan import generate, style_interpolate

start, end = generate('asuka', 2)
style_interpolate(start, end, steps=16)
```

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
