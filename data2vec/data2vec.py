import torch
import torch.nn as nn
import math
from .ema import EMA
from fin2vec import Fin2Vec


class Data2vec(nn.Module):
    TYPES = ["finance"]

    def __init__(self, encoder: nn.Module, model: Fin2Vec, d_model: int, cfg, **kwargs):
        super(Data2vec, self).__init__()
        cfg = cfg["d2v"]["config"]
        self.type_ = cfg["type_"]
        self.encoder = encoder  # PCRN
        self.model = model  # Fin2Vec
        self.__dict__.update(kwargs)

        self.ema = EMA(self.model, cfg)  # EMA acts as the teacher

        self.cfg = cfg
        self.ema_decay = self.cfg["ema_decay"]
        self.ema_end_decay = self.cfg["ema_end_decay"]
        self.ema_anneal_end_step = self.cfg["ema_anneal_end_step"]

        mskRate = 0.2  # p/q
        self.p = int(mskRate * 100)
        self.q = 100
        self.d_model = d_model

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
        studentSrc = torch.tensor(src)
        tknMsk = (
            torch.randint(low=0, high=self.q - 1, size=(srcShape[0], srcShape[1]))
            < self.p
        ) | (~msk)
        studentSrc[tknMsk] = 0

        # student
        x = self.model(studentSrc, idx, msk)

        # teacher
        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(src, idx, msk)

        return x, y, tknMsk
