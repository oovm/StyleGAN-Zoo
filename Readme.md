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

- For jupyter:

![](https://user-images.githubusercontent.com/17541209/71553336-29c55980-2a49-11ea-8e94-18b7ab384706.png)

- For mathematica:

![](https://user-images.githubusercontent.com/17541209/71553454-c5a39500-2a4a-11ea-8513-7d9a475c4c46.png)


## License

| Part      | License                        |
| :-------- | :----------------------------- |
| Code      | [Apache License Version 2.0]() |
| [Asuka]() | [CC0 - Creative Commons]()     |
| [Horo]()  | [CC0 - Creative Commons]()     |
| [Baby]()  | [CC4.0 Non-Commercial]()        |
