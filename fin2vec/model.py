import torch
from torch import nn
import math


# Finance to Vector
class Fin2Vec(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,  # PCRN
        ncodes: int,
        embeddings: (int, int),
        outputs: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int = 7,
        dropout: float = 0.1,
        decoder: nn.Module = None,
    ):
        super(Fin2Vec, self).__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.ncodes = ncodes
        self.embeddings = embeddings[0] * embeddings[1]
        self.word2embedding = nn.Embedding(self.ncodes, self.embeddings)

        self.inputDenseSrc = nn.Linear(self.embeddings, d_model)
        tfENCLayer = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.tfEncoder = nn.Sequential(
            nn.Linear(self.embeddings, d_model),
            nn.ReLU(True),
            nn.TransformerEncoder(tfENCLayer, nlayers),
            nn.Linear(d_model, outputs),
            nn.ReLU(True),
        )

        self.decoder = decoder

    def forward(self, src: torch.tensor, idx: torch.tensor):
        srcShape, idxShape = src.shape, idx.shape
        assert srcShape[0] == idxShape[0] and srcShape[1] == idxShape[1]

        # tokenizing
        src = src.reshape(srcShape[0] * srcShape[1], srcShape[2], srcShape[3])
        with torch.no_grad():
            src = self.encoder(src) * math.sqrt(self.d_model)
        src = src.reshape(srcShape[0], srcShape[1], -1)
        idx = self.word2embedding(idx).reshape(idxShape[0], idxShape[1], -1)
        src += idx

        # transforming
        result = self.tfEncoder(src)

        # decoding
        if self.decoder:
            with torch.no_grad():
                result = self.decoder(result)

        return result
