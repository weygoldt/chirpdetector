# Installation

## Installing Python

This package requires `python>=3.11.5`, which at the time of publishing, is not
installed on all systems. The easiest way of installing a specific python
version isolated from your system environtment is using [`pyenv`](). Follow the
installation instructions and come back.

## Setting the local `python` interpreter

Navigate into your project root directory and run `pyenv local 3.11.5` to set
the local python version to `3.11.5`.

## Creating a virtual environment

Create a virtual environment using `python3 -m venv .venv`. Now activate it by
running `source .venv/bin/activate`.

Alternatively, you can create a virtual environment using `pyenv` directly,
wich has the benefit that it is activated automatically as soon as you `cd`
into the directory. You can do this by running `pyenv virtualenv 3.11.5
chirpdetection` and then running `pyenv local chirpdetection`.

## Cloning the repository

As this project, as well as one of its dependencies, is currently under
development, it is not yet published on `pypi` and you have to install it
from github. To do this, clone this repository, as well as the `gridtools`
repository like so:

```bash
git clone https://github.com/weygoldt/chirpdetector chirpdetector
git clone https://github.com/weygoldt/gridtools gridtools
```

You should now have a `chirpdetector` and a `gridtools` subdirectory in your
project directory.

## Installing with `pip`

Now you just have to install the packages locally using pythons package manager
`pip`. First install `gridtools` and then `chirpdetector`, otherwise
chirpdetector will look for gridtools and exit as it is not installed locally.

```bash
cd gridtools && pip install -e .
cd ../chirpdetector && pip install -e .
```

If both packages and their dependencies installed correctly, you are all set.
You can now continue to [training the detector](training.md) if you want to
train the neural network using your own dataset. In most cases, I recommend to
start by downloading a pretrained model instead.

## Downloading the model

Coming soon
