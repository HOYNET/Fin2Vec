import torch
from torch import nn


class Encoder(nn.Module):
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
            nn.AvgPool1d(kernel_size=160, stride=1),
        )
        self.cnv1DdailySize, self.cnv1DweeklySize, self.cnv1DmonthlySize = (
            self.dates - 4,
            self.dates - 19,
            self.dates - 140,
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
        self.rnnFusion = nn.Linear(self.dates, fusionSize)

        self.finalFusion = nn.Linear(2 * fusionSize, self.embeddingSize[1])

    def forward(self, x):
        cnnInput = x
        rnnInput = x.transpose(-1, -2)
        features = [
            self.cnv1Ddaily(cnnInput),
            self.cnv1Dweekly(cnnInput),
            self.cnv1Dmonthly(cnnInput),
            self.rnn(rnnInput)[0].transpose(-1, -2),
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
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


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
        # self.decoder = Decoder()

    def forward(self, x):
        result = self.encoder(x)
        # result = self.decoder(result)
        return result