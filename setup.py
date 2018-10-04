from distutils.core import setup

setup(
  name = 'smartbee',
  packages = ['smartbee'], # this must be the same as the name above
  version = '0.1.2',
  description = 'Biblioteca para o projeto smartbee',
  author = 'Rhaniel Magalhães',
  author_email = 'rhaniel@alu.ufc.br',
  url = 'https://github.com/rhanielmx/smartbee', # use the URL to the github repo
  install_requires=[
        'bs4==0.0.1',
        'numpy==1.15.2',
        'pandas==0.23.4',
        'matplotlib==3.0.0',
        'requests==2.19.1'
    ],
  classifiers = [
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3'],
)