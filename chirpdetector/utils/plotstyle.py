import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import ListedColormap


def PlotStyle() -> None:
    class style:
        # lightcmap = cmocean.tools.lighten(cmocean.cm.haline, 0.8)

        # units
        cm = 1 / 2.54
        mm = 1 / 25.4

        # colors
        black = "#000000"
        white = "#e0e4f7"
        gray = "#6c6e7d"
        blue = "#89b4fa"
        sapphire = "#74c7ec"
        sky = "#89dceb"
        teal = "#94e2d5"
        green = "#a6e3a1"
        yellow = "#f9d67f"
        orange = "#faa472"
        maroon = "#eb8486"
        purple = "#d89bf7"
        pink = "#f59edb"
        lavender = "#b4befe"

        @classmethod
        def lims(cls, track1, track2):
            """Helper function to get frequency y axis limits from two
            fundamental frequency tracks.

            Args:
                track1 (array): First track
                track2 (array): Second track
                start (int): Index for first value to be plotted
                stop (int): Index for second value to be plotted
                padding (int): Padding for the upper and lower limit

            Returns:
                lower (float): lower limit
                upper (float): upper limit

            """
            allfunds_tmp = (
                np.concatenate(
                    [
                        track1,
                        track2,
                    ]
                )
                .ravel()
                .tolist()
            )
            lower = np.min(allfunds_tmp)
            upper = np.max(allfunds_tmp)
            return lower, upper

        @classmethod
        def circled_annotation(cls, text, axis, xpos, ypos, padding=0.25):
            axis.text(
                xpos,
                ypos,
                text,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(
                    boxstyle=f"circle, pad={padding}",
                    fc="white",
                    ec="black",
                    lw=1,
                ),
            )

        @classmethod
        def fade_cmap(cls, cmap):
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
            my_cmap = ListedColormap(my_cmap)

            return my_cmap

        @classmethod
        def hide_ax(cls, ax):
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
            ax.patch.set_visible(False)

        @classmethod
        def hide_xax(cls, ax):
            ax.xaxis.set_visible(False)
            ax.spines["bottom"].set_visible(False)

        @classmethod
        def hide_yax(cls, ax):
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)

        @classmethod
        def set_boxplot_color(cls, bp, color):
            plt.setp(bp["boxes"], color=color)
            plt.setp(bp["whiskers"], color=white)
            plt.setp(bp["caps"], color=white)
            plt.setp(bp["medians"], color=black)

        @classmethod
        def label_subplots(cls, labels, axes, fig):
            for axis, label in zip(axes, labels):
                X = axis.get_position().x0
                Y = axis.get_position().y1
                fig.text(X, Y, label, weight="bold")

        @classmethod
        def letter_subplots(
            cls, axes=None, letters=None, xoffset=-0.1, yoffset=1.0, **kwargs
        ):
            """Add letters to the corners of subplots (panels). By default each axis is
            given an uppercase bold letter label placed in the upper-left corner.
            Args
                axes : list of pyplot ax objects. default plt.gcf().axes.
                letters : list of strings to use as labels, default ["A", "B", "C", ...]
                xoffset, yoffset : positions of each label relative to plot frame
                (default -0.1,1.0 = upper left margin). Can also be a list of
                offsets, in which case it should be the same length as the number of
                axes.
                Other keyword arguments will be passed to annotate() when panel letters
                are added.
            Returns:
                list of strings for each label added to the axes
            Examples:
                Defaults:
                    >>> fig, axes = plt.subplots(1,3)
                    >>> letter_subplots() # boldfaced A, B, C

                Common labeling schemes inferred from the first letter:
                    >>> fig, axes = plt.subplots(1,4)
                    # panels labeled (a), (b), (c), (d)
                    >>> letter_subplots(letters='(a)')
                Fully custom lettering:
                    >>> fig, axes = plt.subplots(2,1)
                    >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
                Per-axis offsets:
                    >>> fig, axes = plt.subplots(1,2)
                    >>> letter_subplots(axes, xoffset=[-0.1, -0.15])

                Matrix of axes:
                    >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
                    # fig.axes is a list when axes is a 2x2 matrix
                    >>> letter_subplots(fig.axes)
            """

            # get axes:
            if axes is None:
                axes = plt.gcf().axes
            # handle single axes:
            try:
                iter(axes)
            except TypeError:
                axes = [axes]

            # set up letter defaults (and corresponding fontweight):
            fontweight = "bold"
            ulets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(axes)])
            llets = list("abcdefghijklmnopqrstuvwxyz"[: len(axes)])
            if letters is None or letters == "A":
                letters = ulets
            elif letters == "(a)":
                letters = ["({})".format(lett) for lett in llets]
                fontweight = "normal"
            elif letters == "(A)":
                letters = ["({})".format(lett) for lett in ulets]
                fontweight = "normal"
            elif letters in ("lower", "lowercase", "a"):
                letters = llets

            # make sure there are x and y offsets for each ax in axes:
            if isinstance(xoffset, (int, float)):
                xoffset = [xoffset] * len(axes)
            else:
                assert len(xoffset) == len(axes)
            if isinstance(yoffset, (int, float)):
                yoffset = [yoffset] * len(axes)
            else:
                assert len(yoffset) == len(axes)

            # defaults for annotate (kwargs is second so it can overwrite these defaults):
            my_defaults = dict(
                fontweight=fontweight,
                fontsize="large",
                ha="center",
                va="center",
                xycoords="axes fraction",
                annotation_clip=False,
            )
            kwargs = dict(list(my_defaults.items()) + list(kwargs.items()))

            list_txts = []
            for ax, lbl, xoff, yoff in zip(axes, letters, xoffset, yoffset):
                t = ax.annotate(lbl, xy=(xoff, yoff), **kwargs)
                list_txts.append(t)
            return list_txts

        pass

    # rcparams text setup
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    black = "#111116"
    black = "#000000"
    white = "#e0e4f7"
    gray = "#6c6e7d"
    dark_gray = "#2a2a32"

    # rcparams
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["image.cmap"] = "magma"
    plt.rcParams["axes.xmargin"] = 0.05
    plt.rcParams["axes.ymargin"] = 0.1
    plt.rcParams["axes.titlelocation"] = "left"
    plt.rcParams["axes.titlesize"] = BIGGER_SIZE
    # plt.rcParams["axes.titlepad"] = -10
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.loc"] = "best"
    plt.rcParams["legend.borderpad"] = 0.4
    plt.rcParams["legend.facecolor"] = black
    plt.rcParams["legend.edgecolor"] = black
    plt.rcParams["legend.framealpha"] = 0.7
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.fancybox"] = False

    # # specify the custom font to use
    # plt.rcParams["font.family"] = "sans-serif"
    # plt.rcParams["font.sans-serif"] = "Helvetica Now Text"

    # dark mode modifications
    plt.rcParams["boxplot.flierprops.color"] = white
    plt.rcParams["boxplot.flierprops.markeredgecolor"] = gray
    plt.rcParams["boxplot.boxprops.color"] = gray
    plt.rcParams["boxplot.whiskerprops.color"] = gray
    plt.rcParams["boxplot.capprops.color"] = gray
    plt.rcParams["boxplot.medianprops.color"] = black
    plt.rcParams["text.color"] = white
    plt.rcParams["axes.facecolor"] = black  # axes background color
    plt.rcParams["axes.edgecolor"] = white  # axes edge color
    # plt.rcParams["axes.grid"] = True    # display grid or not
    # plt.rcParams["axes.grid.axis"] = "y"  # which axis the grid is applied to
    plt.rcParams["axes.labelcolor"] = white
    plt.rcParams["axes.axisbelow"] = True  # draw axis gridlines and ticks:
    # plt.rcParams["axes.spines.left"] = True  # display axis spines
    # plt.rcParams["axes.spines.bottom"] = True
    # plt.rcParams["axes.spines.top"] = False
    # plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.prop_cycle"] = cycler(
        "color",
        [
            "#89b4fa",
            "#74c7ec",
            "#89dceb",
            "#94e2d5",
            "#a6e3a1",
            "#f9e2af",
            "#fab387",
            "#eba0ac",
            "#f38ba8",
            "#cba6f7",
            "#f5c2e7",
        ],
    )
    plt.rcParams["xtick.color"] = white  # color of the ticks
    plt.rcParams["ytick.color"] = white  # color of the ticks
    plt.rcParams["grid.color"] = white  # grid color
    plt.rcParams["figure.facecolor"] = black  # figure face color
    plt.rcParams["figure.edgecolor"] = black  # figure edge color
    plt.rcParams["savefig.facecolor"] = black  # figure face color when saving

    return style


if __name__ == "__main__":
    s = PlotStyle()

    import matplotlib.cbook as cbook
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2) - Y**2)
    Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    fig1, ax = plt.subplots()
    im = ax.imshow(
        Z,
        interpolation="bilinear",
        cmap=cm.RdYlGn,
        origin="lower",
        extent=[-3, 3, -3, 3],
        vmax=abs(Z).max(),
        vmin=-abs(Z).max(),
    )

    plt.show()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # generate some random test data
    all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

    # plot violin plot
    axs[0].violinplot(all_data, showmeans=False, showmedians=True)
    axs[0].set_title("Violin plot")

    # plot box plot
    axs[1].boxplot(all_data)
    axs[1].set_title("Box plot")

    # adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks(
            [y + 1 for y in range(len(all_data))],
            labels=["x1", "x2", "x3", "x4"],
        )
        ax.set_xlabel("Four separate samples")
        ax.set_ylabel("Observed values")

    plt.show()

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # Compute pie slices
    N = 20
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = 10 * np.random.rand(N)
    width = np.pi / 4 * np.random.rand(N)
    colors = cmo.cm.haline(radii / 10.0)

    ax = plt.subplot(projection="polar")
    ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

    plt.show()

    methods = [
        None,
        "none",
        "nearest",
        "bilinear",
        "bicubic",
        "spline16",
        "spline36",
        "hanning",
        "hamming",
        "hermite",
        "kaiser",
        "quadric",
        "catrom",
        "gaussian",
        "bessel",
        "mitchell",
        "sinc",
        "lanczos",
    ]

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    grid = np.random.rand(4, 4)

    fig, axs = plt.subplots(
        nrows=3,
        ncols=6,
        figsize=(9, 6),
        subplot_kw={"xticks": [], "yticks": []},
    )

    for ax, interp_method in zip(axs.flat, methods):
        ax.imshow(grid, interpolation=interp_method)
        ax.set_title(str(interp_method))

    plt.tight_layout()
    plt.show()
