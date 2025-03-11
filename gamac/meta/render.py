import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.pad_inches'] = 0

COLORS = {
    -1: "gray",
    0: "blue",
    1: "darkorange",
    2: "green",
    3: "purple",
    4: "red",
    5: "black",
    6: "aqua",
    7: "lime",
    8: "gold",
}

def scatter_image(x, y, colors, img_path):
    ax = plt.axes((0, 0, 1, 1), frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    plt.scatter(x, y, marker='.', c=colors, s=1)
    plt.savefig(img_path)
    plt.clf()