"""Detector datasets: the on-disk contract, and builders that write it.

`chirpdetector.data.contract` defines the format every detector dataset
has (manifest, images, YOLO labels, box table with emitter frequencies and
ignore regions, group-wise splits, content hashes), the `DatasetWriter`
builders use, and the validator trainers run before they read anything.
`chirpdetector.data.synthetic` writes tiny datasets of boxes drawn on
noise, for tests.
"""

from chirpdetector.data.contract import (
    BOX_COLUMNS,
    CLASSES,
    SPLITS,
    Dataset,
    DatasetError,
    DatasetWriter,
    ImageRecord,
    Manifest,
    Source,
    Tiling,
    dataset_ground_truth,
    load_dataset,
    read_image,
    validate_dataset,
    write_image,
)

__all__ = [
    "BOX_COLUMNS",
    "CLASSES",
    "SPLITS",
    "Dataset",
    "DatasetError",
    "DatasetWriter",
    "ImageRecord",
    "Manifest",
    "Source",
    "Tiling",
    "dataset_ground_truth",
    "load_dataset",
    "read_image",
    "validate_dataset",
    "write_image",
]
