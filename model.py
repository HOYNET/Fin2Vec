import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, hiddenChnls, layerSize, inputChnls=7, outputChnls=64):
        super().__init__()
        self.cnv1daily = nn.Conv1d(
            in_channels=inputChnls, out_channels=outputChnls, kernel_size=1, stride=1
        )
        self.cnv1weekly = nn.Conv1d(
            in_channels=inputChnls, out_channels=outputChnls, kernel_size=5, stride=1
        )
        self.cnv1monthly = nn.Conv1d(
            in_channels=inputChnls, out_channels=outputChnls, kernel_size=20, stride=1
        )
        self.rnn = nn.GRU(
            input_size=outputChnls,
            hidden_size=hiddenChnls,
            num_layers=layerSize,
            batch_first=True,
        )

    def forward(self, x):
        cnvList = [self.cnv1daily(x), self.cnv1weekly(x), self.cnv1monthly(x)]
        rnnList = []
        for cnv in cnvList:
            rnnList.append(self.rnn(cnv))