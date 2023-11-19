import numpy as np

from chirpdetector.detect_chirps import float_index_interpolation


def test_float_index_interpolation():
    """
    Test the float_index_interpolation function
    """

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
    expected = np.array([1, 2, 3, 3])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Happy path with edge cases
    values = np.array([0, 4])
    expected = np.array([0.5, 4.5])
    actual = float_index_interpolation(values, index_arr, data_arr)
    np.testing.assert_array_equal(actual, expected)

    # Sad path: values outside of index_arr
    values = np.array([-1, 5])
    expected = np.array([np.nan, np.nan])
    np.testing.assert_array_equal(actual, expected)
