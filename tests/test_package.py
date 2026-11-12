import gridtools

import chirpdetector


def test_version_is_v2() -> None:
    assert chirpdetector.__version__.startswith("2.")


def test_uses_gridtools_v2_not_the_pypi_namesake() -> None:
    assert gridtools.__version__.startswith("2.")
