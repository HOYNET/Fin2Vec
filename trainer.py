import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
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

    def train(self, epochs: int, savingPth: str = None) -> None:
        for t in range(epochs):
            print(f"Epoch {t}\n-------------------------------", end=" ")
            trainLoss = self.step()
            testLoss = self.test()
            print(f"Avg Train Loss : {trainLoss} Avg Test Loss : {testLoss}")
            if savingPth:
                path = f"{savingPth}/hoynet_{t}.pth"
                torch.save(self.model.state_dict(), path)

    def step(self) -> float:
        self.model.train()
        trainLoss = 0

        for batch in self.trainLoader:
            src, tgt = batch["src"], batch["tgt"]
            src, tgt = src.to(self.device).to(dtype=torch.float32), tgt.to(
                self.device
            ).to(dtype=torch.float32)

            xMax, yMax = (
                src.max(dim=-1, keepdim=True)[0],
                tgt.max(dim=-1, keepdim=True)[0],
            )
            src, tgt = src / xMax, tgt / yMax

            if torch.isnan(src).any():
                self.replace_nan_with_nearest(src)
            if torch.isnan(tgt).any():
                self.replace_nan_with_nearest(tgt)

            self.optimizer.zero_grad()
            pred = self.model(src.transpose(-1, -2), tgt[:, :, :-1].transpose(-1, -2))
            assert not torch.isnan(pred).any()
            loss = self.lossFn(pred.transpose(-1, -2), tgt[:, :, 1:])
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
                src, tgt = batch["src"], batch["tgt"]
                src, tgt = src.to(self.device).to(dtype=torch.float32), tgt.to(
                    self.device
                ).to(dtype=torch.float32)
                xMax, yMax = (
                    src.max(dim=-1, keepdim=True)[0],
                    tgt.max(dim=-1, keepdim=True)[0],
                )
                src, tgt = src / xMax, tgt / yMax
                if torch.isnan(src).any():
                    self.replace_nan_with_nearest(src)
                if torch.isnan(tgt).any():
                    self.replace_nan_with_nearest(tgt)

                pred = self.model(src)
                test_loss += self.lossFn(pred, tgt).item()

        test_loss /= self.testLength
        return test_loss

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
