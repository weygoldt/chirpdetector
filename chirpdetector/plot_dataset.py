import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.patches import Rectangle


def load_img(path: pathlib.Path) -> np.ndarray:
    img = PIL.Image.open(path)
    img = np.asarray(img)
    return img


def plot_dataset(path: pathlib.Path) -> None:
    labelpath = path / "labels"
    imgpath = path / "images"

    label_paths = list(labelpath.glob("*.txt"))

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

        print(labs)
        print(xmin, ymin, xmax, ymax)

        # plot the image
        fig, ax = plt.subplots()
        ax.imshow(img, origin="lower")
        for i in range(len(xmin)):
            ax.add_patch(
                Rectangle(
                    (xmin[i], ymin[i]),
                    xmax[i] - xmin[i],
                    ymax[i] - ymin[i],
                    fill=False,
                    color="r",
                )
            )
        plt.show()


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="Path to the dataset.",
        required=True,
    )
    args = parser.parse_args()
    plot_dataset(args.path)


def main():
    interface()


if __name__ == "__main__":
    main()
