from torch import nn


class Hoynet(nn.Module):
    def __init__(
        self, inputSize, hiddenSize, nhead, layerSize, dropout, outputSize, device
    ):
        super(Hoynet, self).__init__()
        self.inputDenseSrc = nn.Linear(inputSize, hiddenSize)
        self.inputDenseTgt = nn.Linear(outputSize, hiddenSize)
        self.transformer = nn.Transformer(
            hiddenSize, nhead, layerSize, layerSize, dropout=dropout, batch_first=True
        )
        self.outputDense = nn.Linear(hiddenSize, outputSize)

    def forward(self, src, tgt):
        src = self.inputDenseSrc(src)
        tgt = self.inputDenseTgt(tgt)
        result = self.transformer(src, tgt)
        result = self.outputDense(result)
        return result
