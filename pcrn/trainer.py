import torch
from torch import nn
from .parse import PCRNDataset
from torch.utils.data import DataLoader, random_split

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # for wsl user


class PCRNTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: PCRNDataset,
        batches: int,
        trainRate: float,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lossFn,
        scheduler=None,
    ):
        self.model = model
        self.model.to(device)

        assert trainRate and trainRate < 1 and trainRate > 0
        trainLength = int(len(dataset) * trainRate)
        traindataset, testdataset = random_split(
            dataset, [trainLength, len(dataset) - trainLength]
        )
        self.trainLoader, self.testLoader = (
            DataLoader(
                traindataset,
                batch_size=batches,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            ),
            DataLoader(
                testdataset,
                batch_size=batches,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            ),
        )
        self.trainLength, self.testLength = (
            len(self.trainLoader.dataset),
            len(self.testLoader.dataset),
        )

        self.optimizer = optimizer
        self.lossFn = lossFn
        self.device = device
        self.scheduler = scheduler

    def __call__(self, epochs: int, savingPth: str = None) -> None:
        self.savingPth = savingPth
        for t in range(epochs):
            print(f"Epoch {t} ")
            trainLoss = self.step(t)
            if self.savingPth:
                path = f"{savingPth}/pcrn_{t}.pth"
                torch.save(self.model.state_dict(), path)
            testLoss = self.test(t)
            print(f" Avg Train Loss : {trainLoss:.8f} Avg Test Loss : {testLoss:.8f}")

    def step(self, epoch) -> float:
        self.model.train()
        trainLoss = 0

        for idx, batch in enumerate(self.trainLoader):
            data, label = (
                batch["data"].to(self.device).to(dtype=torch.float32),
                batch["label"].to(self.device).to(dtype=torch.float32),
            )
            data, dmax = self.maxPreProc(data)
            label, lmax = self.maxPreProc(label)

            self.optimizer.zero_grad()
            pred = self.model(data)
            assert not torch.isnan(pred).any()
            loss = self.lossFn(pred, label)
            loss.backward()
            self.optimizer.step()

            if idx % 100 == 0:
                trainLoss *= idx
                trainLoss += loss.item()
                trainLoss /= idx + 1
                print(
                    f"train - {idx} loss : {loss.item()} avg loss : {trainLoss}",
                    end=" ",
                )
                if self.scheduler:
                    self.scheduler.step()
                if self.savingPth:
                    path = f"{self.savingPth}/pcrn_{epoch}_{idx}.pth"
                    torch.save(self.model.state_dict(), path)
                    print(f"Weights Saved on {path} ...", end="")
                print("")
                self.visualize(pred, label, idx)
        return trainLoss

    def test(self, epoch) -> float:
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(self.testLoader):
                data, label = (
                    batch["data"].to(self.device).to(dtype=torch.float32),
                    batch["label"].to(self.device).to(dtype=torch.float32),
                )
                data, dmax = self.maxPreProc(data)
                label, lmax = self.maxPreProc(label)

                pred = self.model(data)
                assert not torch.isnan(pred).any()
                loss = self.lossFn(pred, label)

                if idx % 10 == 0:
                    test_loss *= idx
                    test_loss += loss.item()
                    test_loss /= idx + 1
                    print(f"test - {idx} loss : {test_loss}")

        return test_loss

    def maxPreProc(self, tensor: torch.tensor) -> torch.tensor:
        max = tensor.max(dim=-1, keepdim=True)[0]
        tensor /= max

        if torch.isnan(tensor).any():
            self.replace_nan_with_nearest(tensor)

        return tensor, max

    def visualize(self, pred: torch.tensor, label: torch.tensor, step: int) -> None:
        pred, label = pred[0].detach().cpu().numpy(), label[0].detach().cpu().numpy()
        shape = pred.shape
        fig, ax = plt.subplots(1, shape[-2], figsize=(10 * shape[-2], 10))
        for i in range(shape[-2]):
            ax[i].scatter(
                range(shape[-1]),
                label[i, :],
                label="X",
                color="blue",
                alpha=0.7,
            )
            ax[i].scatter(
                range(shape[-1]),
                pred[i, :],
                label="Prediction",
                color="red",
                alpha=0.7,
            )
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel("Value")
            ax[i].legend()

        # Log the figure to TensorBoard
        plt.savefig(f"pcrn_{step}.png")
        plt.close()

    def replace_nan_with_nearest(self, tensor: torch.tensor) -> torch.tensor:
        if tensor.dim() > 1:
            for i in range(tensor.size(0)):
                tensor[i, :] = self.replace_nan_with_nearest(tensor[i, :])
            return tensor

        isnan = torch.isnan(tensor)
        if torch.all(isnan):
            tensor = tensor.zero_()
            return tensor

        while torch.any(isnan):
            shifted = torch.roll(isnan, 1, dims=0)
            shifted[0] = False
            tensor[isnan] = tensor[shifted]
            isnan = torch.isnan(tensor)

        return tensor
