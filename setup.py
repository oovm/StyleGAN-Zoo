import setuptools
import yaml

meta = yaml.load(open('./meta.yaml'), Loader=yaml.FullLoader)

setuptools.setup(
    name=meta['package']['name'],
    author='aster',
    author_email='galaster@foxmail.com',
    url=meta['source']['url'],
    version=meta['package']['version'],
    description='',

    packages=['sgan'],
    install_requires=[
        # no pytorch
        'matplotlib',
        'numpy',
        'wolframclient'
    ],
)
