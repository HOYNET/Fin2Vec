import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


def visualize(x: np.ndarray, pred: np.ndarray, epoch, batch, columns) -> None:
    xShape, predShape = x.shape, pred.shape
    assert xShape == predShape

    # Create a figure and subplots
    fig, axs = plt.subplots(1, xShape[-2], figsize=(10 * xShape[-1], 10))

    # Loop through each feature and plot it in a separate subplot using line plots
    for i in range(xShape[-2]):
        axs[i].scatter(
            range(xShape[-1]),
            x[i, :],
            label="X",
            color="blue",
            alpha=0.7,
        )
        axs[i].scatter(
            range(xShape[-1]),
            pred[i, :],
            label="Prediction",
            color="red",
            alpha=0.7,
        )
        axs[i].set_title(columns[i])
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Value")
        axs[i].legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig("./checkpoints/" + str(epoch) + "-" + str(batch) + ".png")
    plt.close()

    return None
