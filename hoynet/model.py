from torch import nn
import math


class Hoynet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        inputSize: int,
        outputSize: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        decoder: nn.Module,
        dropout: float = 0.1,
        nlayers: int = 7,
    ):
        super(Hoynet, self).__init__()
        self.encoder = encoder
        self.d_model = d_model

        self.inputDenseSrc = nn.Linear(inputSize, d_model)
        tfENCLayer = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.tfEncoder = nn.Sequential(
            nn.Linear(inputSize, d_model),
            nn.ReLU(True),
            nn.TransformerEncoder(tfENCLayer, nlayers),
            nn.Linear(d_model, outputSize),
            nn.ReLU(True)
        )

        self.decoder = decoder

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        result = self.tfEncoder(src)
        if self.decoder:
            result = self.decoder(result)
        return result
