"""Utility functions for training datasets in the YOLO format."""

import pathlib
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


def load_img(path: pathlib.Path) -> np.ndarray:
    """Load an image from a path as a numpy array.

    Parameters
    ----------
    path : pathlib.Path
        The path to the image.

    Returns
    -------
    img : np.ndarray
        The image as a numpy array.
    """
    img = Image.open(path)
    return np.asarray(img)


def plot_yolo_dataset(path: pathlib.Path, n: int) -> None:
    """Plot n random images YOLO-style dataset.

    Parameters
    ----------
    path : pathlib.Path
        The path to the dataset.

    Returns
    -------
    None
    """
    mpl.use("TkAgg")
    labelpath = path / "labels"
    imgpath = path / "images"

    label_paths = np.array(list(labelpath.glob("*.txt")))
    label_paths = np.random.choice(label_paths, n)

    for lp in label_paths:
        imgp = imgpath / (lp.stem + ".png")
        img = load_img(imgp)
        labs = np.loadtxt(lp, dtype=np.float32).reshape(-1, 5)

        coords = labs[:, 1:]

        # make coords absolute and normalize
        coords[:, 0] *= img.shape[1]
        coords[:, 1] *= img.shape[0]
        coords[:, 2] *= img.shape[1]
        coords[:, 3] *= img.shape[0]

        # turn centerx, centery, width, height into xmin, ymin, xmax, ymax
        xmin = coords[:, 0] - coords[:, 2] / 2
        ymin = coords[:, 1] - coords[:, 3] / 2
        xmax = coords[:, 0] + coords[:, 2] / 2
        ymax = coords[:, 1] + coords[:, 3] / 2

        # plot the image
        _, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
        ax.imshow(img, cmap="magma")
        for i in range(len(xmin)):
            ax.add_patch(
                Rectangle(
                    (xmin[i], ymin[i]),
                    xmax[i] - xmin[i],
                    ymax[i] - ymin[i],
                    fill=False,
                    color="white",
                ),
            )
        ax.set_title(imgp.stem)
        plt.axis("off")
        plt.show()


def clean_yolo_dataset(path: pathlib.Path, img_ext: str) -> None:
    """Remove images and labels when the label file is empty.

    Parameters
    ----------
    path : pathlib.Path
        The path to the dataset.
    img_ext : str

    Returns
    -------
    None
    """
    img_path = path / "images"
    lbl_path = path / "labels"

    images = list(img_path.glob(f"*{img_ext}"))

    for image in images:
        lbl = lbl_path / f"{image.stem}.txt"
        if lbl.stat().st_size == 0:
            image.unlink()
            lbl.unlink()


def subset_yolo_dataset(path: pathlib.Path, img_ext: str, n: int) -> None:
    """Subset a YOLO dataset.

    Parameters
    ----------
    path : pathlib.Path
        The path to the dataset root.
    img_ext : str
        The image extension, e.g. .png or .jpg
    n : int
        The size of the subset

    Returns
    -------
    None
    """
    img_path = path / "images"
    lbl_path = path / "labels"

    images = np.array(img_path.glob(f"*{img_ext}"))
    np.random.shuffle(images)

    images = images[:n]

    subset_dir = path.parent / f"{path.name}_subset"
    subset_dir.mkdir(exist_ok=True)

    subset_img_path = subset_dir / "images"
    subset_img_path.mkdir(exist_ok=True)
    subset_lbl_path = subset_dir / "labels"
    subset_lbl_path.mkdir(exist_ok=True)

    shutil.copy(path / "classes.txt", subset_dir)

    for image in images:
        shutil.copy(image, subset_img_path)
        shutil.copy(lbl_path / f"{image.stem}.txt", subset_lbl_path)


def merge_yolo_datasets(
    dataset1: pathlib.Path,
    dataset2: pathlib.Path,
    output: pathlib.Path,
) -> None:
    """Merge two yolo-style datasets into one.

    Parameters
    ----------
    dataset1 : str
        The path to the first dataset.
    dataset2 : str
        The path to the second dataset.
    output : str
        The path to the output dataset.

    Returns
    -------
    None
    """
    dataset1 = pathlib.Path(dataset1)
    dataset2 = pathlib.Path(dataset2)
    output = pathlib.Path(output)

    if not dataset1.exists():
        msg = f"{dataset1} does not exist."
        raise FileNotFoundError(msg)
    if not dataset2.exists():
        msg = f"{dataset2} does not exist."
        raise FileNotFoundError(msg)
    if output.exists():
        msg = f"{output} already exists."
        raise FileExistsError(msg)

    output_images = output / "images"
    output_images.mkdir(parents=True, exist_ok=False)
    output_labels = output / "labels"
    output_labels.mkdir(parents=True, exist_ok=False)

    imgs1 = list((dataset1 / "images").iterdir())
    labels1 = list((dataset1 / "labels").iterdir())
    imgs2 = list((dataset2 / "images").iterdir())
    labels2 = list((dataset2 / "labels").iterdir())

    print(f"Found {len(imgs1)} images in {dataset1}.")
    print(f"Found {len(imgs2)} images in {dataset2}.")

    print(f"Copying images and labels to {output}...")
    for idx, _ in enumerate(imgs1):
        shutil.copy(imgs1[idx], output_images / imgs1[idx].name)
        shutil.copy(labels1[idx], output_labels / labels1[idx].name)

    for idx, _ in enumerate(imgs2):
        shutil.copy(imgs2[idx], output_images / imgs2[idx].name)
        shutil.copy(labels2[idx], output_labels / labels2[idx].name)

    classes = dataset1 / "classes.txt"
    shutil.copy(classes, output / classes.name)

    print(f"Done. Merged {len(imgs1) + len(imgs2)} images.")
