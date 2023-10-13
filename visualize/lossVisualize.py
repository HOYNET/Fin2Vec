import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Get Path of Files.")
parser.add_argument(
    "-f",
    "--lossFile",
    metavar="path",
    dest="lossFile",
    type=str,
    help="File Contains Loss Info",
)

parser.add_argument(
    "-i",
    "--lossInfo",
    metavar="path",
    dest="lossInfo",
    type=str,
    help="Loss Info",
)


# 파일 읽기 함수
def read_losses(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    train_losses = []
    test_losses = []
    for line in lines:
        if "Train Loss" in line:
            train_losses.append(float(line.split(":")[1].strip()))
        elif "Test  Loss" in line:
            test_losses.append(float(line.split(":")[1].strip()))

    if len(train_losses) != len(test_losses):
        length = min(len(train_losses), len(test_losses))
        train_losses = train_losses[:length]
        test_losses = test_losses[:length]
    return train_losses, test_losses


# 그래프 그리기 함수
def plot_losses(info, train_losses, test_losses):
    epochs = list(range(1, len(train_losses) + 1))

    plt.plot(epochs, train_losses, marker="^", label="Train Loss")
    plt.plot(epochs, test_losses, marker="s", label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train and Test Loss Over Epochs : {info}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{info}.png")


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.lossFile and args.lossInfo

    # 파일 경로 지정
    file_path = args.lossFile

    # 파일에서 데이터를 읽어옵니다.
    train_losses, test_losses = read_losses(file_path)

    # 읽어온 손실 데이터로 그래프를 그립니다.
    plot_losses(args.lossInfo, train_losses, test_losses)
