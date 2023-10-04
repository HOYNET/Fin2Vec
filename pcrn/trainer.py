import torch
from torch import nn
from .parse import PCRNDataset
from torch.utils.data import DataLoader, random_split


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
    ):
        self.model = model
        self.model.to(device)

        assert trainRate and trainRate < 1 and trainRate > 0
        trainLength = int(len(dataset) * trainRate)
        traindataset, testdataset = random_split(
            dataset, [trainLength, len(dataset) - trainLength]
        )
        self.trainLoader, self.testLoader = DataLoader(
            traindataset,
            batch_size=batches,
            shuffle=True,
        ), DataLoader(testdataset, batch_size=batches, shuffle=True)
        self.trainLength, self.testLength = len(self.trainLoader.dataset), len(
            self.testLoader.dataset
        )

        self.optimizer = optimizer
        self.lossFn = lossFn
        self.device = device

    def __call__(self, epochs: int, savingPth: str = None) -> None:
        for t in range(epochs):
            print(f"Epoch {t} ", end=" ")
            trainLoss = self.step()
            testLoss = self.test()
            print(f" Avg Train Loss : {trainLoss:.8f} Avg Test Loss : {testLoss:.8f}")
            if savingPth:
                path = f"{savingPth}/hoynet_{t}.pth"
                torch.save(self.model.state_dict(), path)

    def step(self) -> float:
        self.model.train()
        trainLoss = 0

        for batch in self.trainLoader:
            print("#", end="")
            data, label = batch["data"].to(self.device).to(dtype=torch.float32), batch[
                "label"
            ].to(self.device).to(dtype=torch.float32)
            data, label = self.maxPreProc(data), self.maxPreProc(label)

            self.optimizer.zero_grad()
            pred = self.model(data)
            assert not torch.isnan(pred).any()
            loss = self.lossFn(pred, label)
            loss.backward()
            self.optimizer.step()
            trainLoss += loss.item()

        trainLoss /= self.trainLength
        return trainLoss

    def test(self) -> float:
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in self.testLoader:
                print("#", end="")
                data, label = batch["data"].to(self.device).to(
                    dtype=torch.float32
                ), batch["label"].to(self.device).to(dtype=torch.float32)
                data, label = self.maxPreProc(data), self.maxPreProc(label)

                pred = self.model(data)
                assert not torch.isnan(pred).any()
                test_loss += self.lossFn(pred, label)

        test_loss /= self.testLength
        return test_loss

    def maxPreProc(self, tensor: torch.tensor) -> torch.tensor:
        max = tensor.max(dim=-1, keepdim=True)[0]
        tensor /= max

        if torch.isnan(tensor).any():
            self.replace_nan_with_nearest(tensor)

        return tensor

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
