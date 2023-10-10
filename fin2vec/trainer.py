import torch
from omegaconf import DictConfig
from .parse import Fin2VecDataset
from torch.utils.data import DataLoader, random_split
from data2vec import Data2vec
from .model import Fin2Vec


class Fin2VecTrainer:
    def __init__(
        self,
        data2vec: Data2vec,
        model: Fin2Vec,
        dataset: Fin2VecDataset,
        batches: int,
        trainRate: float,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        d2v_cfg: DictConfig,
        lossFn,
    ):
        self.data2vec, self.model = data2vec, model
        self.model.to(device)
        self.data2vec.to(device)

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
            ),
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
                torch.save(self.model.state_dict(), path)

    def step(self) -> float:
        self.model.train()
        trainLoss = 0

        for i, batch in enumerate(self.trainLoader):
            src, idx, msk = (
                self.maxPreProc(batch["src"].to(self.device).to(dtype=torch.float32)),
                batch["index"].to(self.device).to(dtype=torch.int32),
                batch["mask"].to(self.device).to(dtype=torch.bool),
            )

            self.optimizer.zero_grad()
            pred, tgt, mask = self.data2vec(src, idx, msk)
            assert not (torch.isnan(pred).any() and torch.isnan(tgt).any())

            mask = mask.to(dtype=torch.bool)
            loss = self.lossFn(pred[mask], tgt[mask])
            loss.backward()
            self.optimizer.step()
            print(f"Train Batch {i}  Loss : {loss.item()}")
            trainLoss += loss.item()

        trainLoss /= self.trainLength
        return trainLoss

    def test(self) -> float:
        self.model.eval()
        test_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(self.testLoader):
                src, idx, msk = (
                    self.maxPreProc(
                        batch["src"].to(self.device).to(dtype=torch.float32)
                    ),
                    batch["index"].to(self.device).to(dtype=torch.int32),
                    batch["mask"].to(self.device).to(dtype=torch.bool),
                )

                pred, tgt, mask = self.data2vec(src, idx, msk)
                assert not (torch.isnan(pred).any() and torch.isnan(tgt).any())
                loss = self.lossFn(pred[mask], tgt[mask])

                print(f"Test  Batch {i}  Loss : {loss.item()}")
                test_loss += loss.item()

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
