"""Tests for the detect_chirps module."""

import numpy as np
import torch

from chirpdetector.detect_chirps import (
    coords_to_mpl_rectangle,
    float_index_interpolation,
    spec_to_image,
)


def test_float_index_interpolation() -> None:
    """Test the float_index_interpolation function."""
    index_arr = np.arange(5)
    data_arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    # Happy path single value
    values = np.array([1.5])
    expected = np.array([2])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Happy path multiple values
    values = np.array([1.5, 2.5, 3.5])
    expected = np.array([2, 3, 4])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Happy path multiple values with duplicates
    values = np.array([1.5, 2.5, 3.5, 3.5])
    expected = np.array([2, 3, 4, 4])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Happy path with integers
    values = np.array([1, 2, 3, 3])
    expected = np.array([1.5, 2.5, 3.5, 3.5])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Happy path with edge cases
    values = np.array([0, 4])
    expected = np.array([0.5, 4.5])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Happy path: values not sorted
    values = np.array([2, 1])
    expected = np.array([2.5, 1.5])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Sad path: values outside of index_arr
    values = np.array([-2, 6])
    try:
        float_index_interpolation(values, index_arr, data_arr)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError"
        raise AssertionError(msg)


def test_coords_to_mpl_rectangle() -> None:
    """Test the corner_coords_to_center_coords function."""
    # Happy path
    corner_coords = np.array([[1, 1, 2, 2]])
    expected = np.array([[1, 1, 1, 1]])
    actual = coords_to_mpl_rectangle(corner_coords)
    np.testing.assert_array_equal(actual, expected)

    # Happy path with multiple values
    corner_coords = np.array([[1, 1, 2, 2], [2, 2, 3, 3]])
    expected = np.array([[1, 1, 1, 1], [2, 2, 1, 1]])
    actual = coords_to_mpl_rectangle(corner_coords)
    np.testing.assert_array_equal(actual, expected)

    # Sad path: wrong number of columns
    corner_coords = np.array([[1, 1, 2, 2, 3]])
    try:
        coords_to_mpl_rectangle(corner_coords)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError"
        raise AssertionError(msg)

    # Sad path: wrong dimensions
    corner_coords = np.array([1, 1, 2, 2])
    try:
        coords_to_mpl_rectangle(corner_coords)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError"
        raise AssertionError(msg)


def test_spec_to_image() -> None:
    """Test the spec_to_image function."""
    # Happy path
    spec = torch.tensor(np.array([[1, 1, 1], [2, 2, 2]]))
    expected = (
        torch.tensor([[0, 0, 0], [1, 1, 1]])
        .repeat(3, 1, 1)
        .view(3, 2, 3)
        .float()
    )
    actual = spec_to_image(spec)
    np.testing.assert_array_equal(actual, expected)

    # Sad path: No data
    spec = torch.tensor(np.array([[1, 1, 1], [1, 1, 1]]))
    try:
        spec_to_image(spec)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError"
        raise AssertionError(msg)

    # Sad path: Wrong dimensions
    spec = torch.tensor(np.array([1, 1, 1]))
    try:
        spec_to_image(spec)
    except ValueError:
        pass
    else:
        msg = "Expected ValueError"
        raise AssertionError(msg)

    # Sad path: Wrong type:
    spec = np.array([[1, 1, 1], [1, 1, 1]])
    try:
        spec_to_image(spec)  # type: ignore
    except TypeError:
        pass
    else:
        msg = "Expected TypeError"
        raise AssertionError(msg)
