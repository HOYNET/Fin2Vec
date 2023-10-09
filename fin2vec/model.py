import torch
from torch import nn
import math
import random


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
        self.ncodes = ncodes + 1
        print(embeddings)
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

    def forward(
        self, student: bool, src: torch.tensor, idx: torch.tensor, end: torch.tensor
    ):
        srcShape, idxShape = src.shape, idx.shape
        assert srcShape[0] == idxShape[0] and srcShape[1] == idxShape[1]

        # tokenizing
        src = src.reshape(srcShape[0] * srcShape[1], srcShape[2], srcShape[3])
        with torch.no_grad():
            src = self.encoder(src) * math.sqrt(self.d_model)
        src = src.reshape(srcShape[0], srcShape[1], -1)

        mask = torch.Tensor(srcShape[0], srcShape[1]).fill_(False)
        if student:
            for batch_ in range(srcShape[0]):
                ##############여기에 num_random에 진자 데이터 개수를 적기 end값을 적으면 될듯함##################
                num_random = srcShape[1]
                maskedIDX_list = random.sample(
                    range(num_random), int(num_random * 15 / 100)
                )
                src[batch_, maskedIDX_list, :] = 0
                idx[batch_, maskedIDX_list] = self.masked_code
                mask[batch_, maskedIDX_list] = True

        idx = self.word2embedding(idx.to(torch.int64)).reshape(
            idxShape[0], idxShape[1], -1
        )
        src += idx

        # transforming
        src = self.tfLinear0(src)
        result = self.tfEncoder(src, src_key_padding_mask=end)
        result = self.tfLinear1(result)

        # decoding
        if self.decoder:
            with torch.no_grad():
                result = self.decoder(result)

        return result, mask
