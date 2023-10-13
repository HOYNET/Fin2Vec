import torch
import torch.nn as nn
import math
from .ema import EMA
import yaml


class Data2vec(nn.Module):
    TYPES = ["finance"]

    def __init__(
        self,
        encoder: nn.Module,
        model: nn.Module,
        d_model: int,
        decay,
        device,
        endDecay,
        endStep,
        **kwargs
    ):
        super(Data2vec, self).__init__()
        self.encoder = encoder  # PCRN
        self.model = model  # Fin2Vec
        self.__dict__.update(kwargs)

        self.ema = EMA(self.model, decay, device)  # EMA acts as the teacher

        self.ema_decay = decay
        self.ema_end_decay = endDecay
        self.ema_anneal_end_step = endStep

        mskRate = 0.2  # p/q
        self.p = int(mskRate * 100)
        self.q = 100
        self.d_model = d_model

        self.device = torch.device(device)

    def ema_step(self):
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, src, idx, msk):
        # tokenizing
        srcShape, idxShape = src.shape, idx.shape
        assert srcShape[0] == idxShape[0] and srcShape[1] == idxShape[1]
        src = src.reshape(srcShape[0] * srcShape[1], srcShape[2], srcShape[3])
        with torch.no_grad():
            src = self.encoder(src) * math.sqrt(self.d_model)
        src = src.reshape(srcShape[0], srcShape[1], -1)

        # masking
        # studentSrc = torch.tensor(src)
        studentSrc = src.clone().detach().to(self.device)
        tknMsk = (
            torch.randint(
                low=0,
                high=self.q - 1,
                size=(srcShape[0], srcShape[1]),
                device=self.device,
            )
            < self.p
        ) & msk
        studentSrc[tknMsk] = 0

        # student
        x = self.model(studentSrc, idx, msk)

        # teacher
        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(src, idx, msk)
            self.ema.step(self.model)

        return x, y, tknMsk


def Cofing2Data2Vec(path, model: nn.Module, encoder: nn.Module, device) -> Data2vec:
    with open(path) as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        data2vec = yml["data2vec"]

    model = Data2vec(
        encoder,
        model,
        model.d_model,
        data2vec["ema_decay"],
        device,
        data2vec["ema_end_decay"],
        data2vec["ema_anneal_end_step"],
    )

    model.to(device)

    return model
