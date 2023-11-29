"""Test the convert_data module."""

import numpy as np
from PIL import Image
import pathlib

from chirpdetector.convert_data import make_file_tree, numpy_to_pil


def test_make_file_tree(tmp_path: pathlib.Path) -> None:
    """Test the make_file_tree function."""
    # Happy path non-existing directory
    make_file_tree(tmp_path)
    assert (tmp_path / "images").exists()
    assert (tmp_path / "labels").exists()

    # Happy path existing directory
    make_file_tree(tmp_path)
    make_file_tree(tmp_path)
    assert (tmp_path / "images").exists()
    assert (tmp_path / "labels").exists()

    # Sad path with str
    try:
        make_file_tree(str(tmp_path)) # type: ignore
    except TypeError:
        pass
    else:
        msg = "TypeError not raised"
        raise AssertionError(msg)


def test_numpy_to_pil() -> None:
    """Test the numpy_to_pil function."""
    # Happy path
    array = np.zeros((10, 10), dtype=np.float32)
    array[0, 0] = 0.5
    img = numpy_to_pil(array)
    assert isinstance(img, Image.Image)
    assert img.size == (10, 10)
    img_min, img_max = 0, 255
    assert np.min(np.asarray(img)) == img_min
    assert np.max(np.asarray(img)) == img_max

    # Sad path with wrong shape
    array = np.zeros((10, 10, 10))
    try:
        img = numpy_to_pil(array)
    except ValueError:
        pass
    else:
        msg = "ValueError not raised"
        raise AssertionError(msg)

    # Sad path without data
    array = np.zeros((0, 0))
    try:
        img = numpy_to_pil(array)
    except ValueError:
        pass
    else:
        msg = "ValueError not raised"
        raise AssertionError(msg)
