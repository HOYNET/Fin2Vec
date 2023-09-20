import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        dates: int,  # number of sequences
        inputSize: int,  # number of features
        hiddenSize: int,  # size of hiddenlayer
        layerSize: int,  # size of layer in GRU
        fusionSize: int,  # size of fusion features
        embeddingSize: (int, int),  # size of embedding
    ):
        super().__init__()
        self.dates = dates
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.layerSize = layerSize
        self.fusionSize = fusionSize
        self.embeddingSize = embeddingSize
        self.cnv1Ddaily = nn.Sequential(
            nn.Conv1d(
                in_channels=inputSize,
                out_channels=hiddenSize,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="replicate",
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hiddenSize,
                out_channels=self.embeddingSize[0],
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="replicate",
            ),
            nn.AvgPool1d(kernel_size=5, stride=1),
        )
        self.cnv1Dweekly = nn.Sequential(
            nn.Conv1d(
                in_channels=inputSize,
                out_channels=hiddenSize,
                kernel_size=5,
                stride=1,
                padding=2,
                padding_mode="replicate",
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hiddenSize,
                out_channels=self.embeddingSize[0],
                kernel_size=5,
                stride=1,
                padding=2,
                padding_mode="replicate",
            ),
            nn.AvgPool1d(kernel_size=20, stride=1),
        )
        self.cnv1Dmonthly = nn.Sequential(
            nn.Conv1d(
                in_channels=inputSize,
                out_channels=hiddenSize,
                kernel_size=11,
                stride=1,
                padding=10,
                padding_mode="replicate",
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hiddenSize,
                out_channels=self.embeddingSize[0],
                kernel_size=10,
                stride=1,
                padding=9,
                padding_mode="replicate",
            ),
            nn.AvgPool1d(kernel_size=self.dates, stride=1),
        )
        self.cnv1DdailySize, self.cnv1DweeklySize, self.cnv1DmonthlySize = (
            self.dates - 4,
            self.dates - 19,
            20,
        )
        self.cnnFusion = nn.Linear(
            self.cnv1DdailySize + self.cnv1DweeklySize + self.cnv1DmonthlySize,
            fusionSize,
        )

        self.rnn = nn.Sequential(
            nn.GRU(
                input_size=self.inputSize,
                hidden_size=self.embeddingSize[0],
                num_layers=self.layerSize,
                batch_first=True,
            )
        )
        self.rnnFusion = nn.Linear(self.layerSize, fusionSize)

        self.finalFusion = nn.Sequential(
            nn.Linear(2 * fusionSize, self.embeddingSize[1]),
            nn.BatchNorm1d(embeddingSize[0]),
        )

    def forward(self, x):
        cnnInput = x
        rnnInput = x.transpose(-1, -2)
        features = [
            self.cnv1Ddaily(cnnInput),
            self.cnv1Dweekly(cnnInput),
            self.cnv1Dmonthly(cnnInput),
            self.rnn(rnnInput)[1].transpose(-2, -3).transpose(-1, -2),
        ]

        cnnFusion = torch.concat(features[0:3], dim=2)
        cnnFusion = self.cnnFusion(cnnFusion)
        rnnFusion = self.rnnFusion(features[-1])

        finalFusion = torch.concat((cnnFusion, rnnFusion), dim=2)
        embedding = self.finalFusion(finalFusion)

        return embedding

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )


class Decoder(nn.Module):
    def __init__(self, dates: int, outputSize: int, embeddingSize: (int, int)):
        super().__init__()
        self.dates = dates
        self.outputSize = outputSize
        self.embeddingSize = embeddingSize

        self.larger0 = nn.Sequential(
            nn.Linear(embeddingSize[1], embeddingSize[1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=self.embeddingSize[0],
                out_channels=self.embeddingSize[0] * 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Linear(embeddingSize[1] * 2, embeddingSize[1] * 4),
        )

        self.cnv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embeddingSize[0]*2,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=64,
                out_channels=outputSize,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.larger1 = nn.Linear(embeddingSize[1] * 4, dates)

    def forward(self, x):
        result = self.larger0(x)
        result = self.cnv(result)
        result = self.larger1(result)
        return result


class Hoynet(nn.Module):
    def __init__(
        self,
        dates,
        inputSize,
        hiddenSize,
        layerSize,
        fusionSize,
        embeddingSize: (int, int),
    ):
        super().__init__()
        self.encoder = Encoder(
            dates, inputSize, hiddenSize, layerSize, fusionSize, embeddingSize
        )
        self.decoder = Decoder(dates, inputSize, embeddingSize)

    def forward(self, x):
        result = self.encoder(x)
        result = self.decoder(result)
        return result
