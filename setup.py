import setuptools

setuptools.setup(
    name='stylegan_zoo',
    author='aster',
    author_email='galaster@foxmail.com',
    url='https://github.com/GalAster/StyleGAN-Zoo',
    version='0.12.4',
    description='none',

    packages=['sgan'],
    install_requires=[
        # no pytorch
        'matplotlib',
        'numpy',
        'wolframclient'
    ],
)
