import torch
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
        testRate: float,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lossFn,
    ):
        self.data2vec, self.model = data2vec, model
        self.model.to(device)
        self.data2vec.to(device)

        assert testRate and testRate < 1 and testRate > 0
        testLength = int(len(dataset) * testRate)
        traindataset, testdataset = random_split(
            dataset, [len(dataset) - testLength, testLength]
        )
        self.trainLoader, self.testLoader = (
            DataLoader(
                traindataset,
                batch_size=batches,
                shuffle=True,
                pin_memory=True,
                num_workers=5,
            ),
            DataLoader(
                testdataset,
                batch_size=batches,
                shuffle=True,
                pin_memory=True,
                num_workers=5,
            ),
        )
        self.trainLength, self.testLength = (
            len(self.trainLoader.dataset),
            len(self.testLoader.dataset),
        )

        self.optimizer = optimizer
        self.lossFn = lossFn
        self.device = device

    def __call__(self, epochs: int, savingPth: str = None) -> None:
        for t in range(epochs):
            print(f"Epoch {t}")
            trainLoss = self.step()
            print(f"Avg Train Loss : {trainLoss}")
            testLoss = self.test()
            print(f"Avg Test  Loss : {testLoss}")
            if savingPth:
                path = f"{savingPth}/hoynet_{t}.pth"
                torch.save(self.model.state_dict(), path)

    def step(self) -> float:
        self.model.train()
        n = 0
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

            trainLoss += loss.item()
            n += 1

        trainLoss /= n
        return trainLoss

    def test(self) -> float:
        self.model.eval()
        n = 0
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

                test_loss += loss.item()
                n += 1

        test_loss /= n
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
