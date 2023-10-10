import torch
from torch import nn


# Finance to Vector
class Fin2Vec(nn.Module):
    def __init__(
        self,
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
        self.d_model = d_model
        self.ncodes = ncodes + 1
        self.embeddings = embeddings[0] * embeddings[1]
        self.masked_code = ncodes
        self.word2embedding = nn.Embedding(self.ncodes, self.embeddings)

        self.inputDenseSrc = nn.Linear(self.embeddings, d_model)
        tfENCLayer = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )

        self.tfLinear0 = nn.Sequential(
            nn.Linear(self.embeddings, d_model), nn.ReLU(True),
        )
        self.tfEncoder = nn.TransformerEncoder(tfENCLayer, nlayers)
        self.tfLinear1 = nn.Sequential(nn.Linear(d_model, outputs), nn.ReLU(True),)

        self.decoder = decoder

    def forward(self, src: torch.tensor, idx: torch.tensor, msk: torch.tensor):
        srcShape, idxShape = src.shape, idx.shape

        src += self.word2embedding(idx.to(torch.int64)).reshape(
            idxShape[0], idxShape[1], -1
        )

        # transforming
        src = self.tfLinear0(src)
        result = self.tfEncoder(src, src_key_padding_mask=msk)
        result = self.tfLinear1(result)

        # decoding
        if self.decoder:
            with torch.no_grad():
                result = self.decoder(result)

        return result
