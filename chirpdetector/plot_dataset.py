import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.patches import Rectangle


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
    img = PIL.Image.open(path)
    img = np.asarray(img)
    return img


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
    matplotlib.use("TkAgg")
    labelpath = path / "labels"
    imgpath = path / "images"

    label_paths = list(labelpath.glob("*.txt"))
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
                )
            )
        ax.set_title(imgp.stem)
        plt.axis("off")
        plt.show()


def plot_yolo_dataset_cli(path: str, n: int) -> None:
    path = pathlib.Path(path)
    plot_yolo_dataset(path, n)
