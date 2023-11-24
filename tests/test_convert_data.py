#!/usr/bin/env python3

"""Test the convert_data module."""

import numpy as np
from PIL import Image

from chirpdetector.convert_data import make_file_tree, numpy_to_pil


def test_make_file_tree(tmp_path):
    """
    Test the make_file_tree function
    """
    # Happy path non-existing directory
    make_file_tree(tmp_path)
    assert (tmp_path / "images").exists()
    assert (tmp_path / "labels").exists()

    # Happy path existing directory
    make_file_tree(tmp_path)
    make_file_tree(tmp_path)
    assert (tmp_path / "images").exists()
    assert (tmp_path / "labels").exists()

    # Happy path with str
    make_file_tree(str(tmp_path))
    assert (tmp_path / "images").exists()
    assert (tmp_path / "labels").exists()


def test_numpy_to_pil():
    """
    Test the numpy_to_pil function
    """
    # Happy path
    array = np.zeros((10, 10), dtype=np.float32)
    array[0, 0] = 0.5
    img = numpy_to_pil(array)
    assert isinstance(img, Image.Image)
    assert img.size == (10, 10)
    assert np.min(np.asarray(img)) == 0
    assert np.max(np.asarray(img)) == 255

    # Sad path with wrong shape
    array = np.zeros((10, 10, 10))
    try:
        img = numpy_to_pil(array)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised")

    # Sad path without data
    array = np.zeros((0, 0))
    try:
        img = numpy_to_pil(array)
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError not raised")
