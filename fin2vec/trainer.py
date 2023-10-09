import torch
from torch import nn
from omegaconf import DictConfig
from .parse import Fin2VecDataset
from torch.utils.data import DataLoader, random_split


class Fin2VecTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        model: nn.Module,
        dataset: Fin2VecDataset,
        batches: int,
        trainRate: float,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        d2v_cfg: DictConfig,
        lossFn,
    ):
        self.encoder = encoder  # fin2vec
        self.model = model  # data2vec
        self.model.to(device)

        assert trainRate and trainRate < 1 and trainRate > 0
        trainLength = int(len(dataset) * trainRate)
        traindataset, testdataset = random_split(
            dataset, [trainLength, len(dataset) - trainLength]
        )
        self.trainLoader, self.testLoader = (
            DataLoader(traindataset, batch_size=batches, shuffle=True,),
            DataLoader(testdataset, batch_size=batches, shuffle=True),
        )
        self.trainLength, self.testLength = (
            len(self.trainLoader.dataset),
            len(self.testLoader.dataset),
        )

        self.optimizer = optimizer
        self.lossFn = lossFn
        self.device = device
        self.d2v_cfg = d2v_cfg

    def __call__(self, epochs: int, savingPth: str = None) -> None:
        for t in range(epochs):
            print(f"Epoch {t} ", end=" ")
            trainLoss = self.step()
            testLoss = self.test()
            print(f" Avg Train Loss : {trainLoss:.8f} Avg Test Loss : {testLoss:.8f}")
            if savingPth:
                path = f"{savingPth}/hoynet_{t}.pth"
                torch.save(self.encoder.state_dict(), path)

    def step(self) -> float:
        self.model.train()
        trainLoss = 0

        for batch in self.trainLoader:
            print("#", end="")
            src, idx, end = (
                batch["src"].to(self.device).to(dtype=torch.float32),
                batch["index"].to(self.device).to(dtype=torch.float32),
                batch["end"].to(self.device).to(dtype=torch.bool),
            )
            src, idx = self.maxPreProc(src), self.maxPreProc(idx)

            self.optimizer.zero_grad()
            pred, tgt, mask = self.model(
                encoder=self.encoder, cfg=self.d2v_cfg, src=src, idx=idx, end=end
            )  # student outputs, teacher outputs in order
            assert not (torch.isnan(pred).any() and torch.isnan(tgt).any())
            mask = mask.to(dtype=torch.bool)
            loss = self.lossFn(pred[mask], tgt[mask])
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
                src, idx = (
                    batch["src"].to(self.device).to(dtype=torch.float32),
                    batch["index"].to(self.device).to(dtype=torch.float32),
                )
                src, idx = self.maxPreProc(src), self.maxPreProc(idx)

                pred, tgt, mask = self.model(
                    encoder=self.encoder, cfg=self.d2v_cfg, src=src, idx=idx
                )  # student outputs, teacher outputs in order
                assert not torch.isnan(pred).any() and torch.isnan(tgt).any()
                test_loss += self.lossFn(pred[mask], tgt[mask])

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
