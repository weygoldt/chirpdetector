#!/usr/bin/env python3

"""Test the configfiles module."""

from chirpdetector.utils.configfiles import copy_config, load_config


def test_copy_config(tmp_path):
    """
    Test the copy_config function
    """
    copy_config(tmp_path)
    assert (tmp_path / "chirpdetector.toml").exists()


def test_load_config(tmp_path):
    """
    Test the load_config function
    """
    copy_config(tmp_path)
    load_config(tmp_path / "chirpdetector.toml")
    assert (tmp_path / "chirpdetector.toml").exists()
