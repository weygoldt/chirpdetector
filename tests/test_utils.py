#!/usr/bin/env python3

"""Test the gridtools.models.utils module."""

import torch

from chirpdetector.models.utils import (
    collate_fn,
    get_device,
    get_transforms,
    load_fasterrcnn,
)


def test_get_device():
    """
    Test the get_device function
    """
    device = str(get_device())
    assert device == "cuda" or device == "cpu" or device == "mps"


def test_get_transforms():
    """
    Test the get_transforms function
    """

    # Happy path train
    transforms = get_transforms(10, 10, True)
    assert transforms is not None

    # Happy path test
    transforms = get_transforms(10, 10, False)
    assert transforms is not None

    # Sad path wrong width
    try:
        transforms = get_transforms(10.0, 10, True)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected TypeError")

    # Sad path wrong height
    try:
        transforms = get_transforms(10, 10.0, True)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected TypeError")

    # Sad path wrong train
    try:
        transforms = get_transforms(10, 10, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected TypeError")


def test_collate_fn():
    """
    Test the collate_fn function
    """

    input = [[1, 1, 1], [2, 2, 2]]
    expected = ((1, 2), (1, 2), (1, 2))
    actual = collate_fn(input)
    assert actual == expected


def test_load_fasterrcnn():
    """
    Test the load_fasterrcnn function
    """

    # Happy path
    model = load_fasterrcnn(2)
    assert model is not None
    assert isinstance(model, torch.nn.Module)

    # Sad path
    try:
        model = load_fasterrcnn(2.0)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError")
