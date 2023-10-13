import torch
from torch import nn
import yaml


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

        self.kernelSizes = self.getKernelSize(self.dates)
        self.cnv1Ds: nn.ModuleList = nn.ModuleList()
        self.cnv1DSizes: list = []
        for kernelSize in self.kernelSizes:
            cnv: nn.Sequential = None
            cnv = nn.Sequential(
                nn.Conv1d(
                    in_channels=inputSize,
                    out_channels=hiddenSize,
                    kernel_size=kernelSize,
                    stride=1,
                    padding=0,
                    padding_mode="replicate",
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=hiddenSize,
                    out_channels=self.embeddingSize[0],
                    kernel_size=kernelSize,
                    stride=1,
                    padding=0,
                    padding_mode="replicate",
                ),
                nn.AvgPool1d(kernel_size=5, stride=1),
            )
            self.cnv1Ds.append(cnv)
            self.cnv1DSizes.append(self.dates - 2 * kernelSize - 2)

        self.cnnFusion = nn.Linear(
            sum(self.cnv1DSizes),
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
        features: list[torch.Tensor] = list(map(lambda m: m(cnnInput), self.cnv1Ds))
        features.append(self.rnn(rnnInput)[1].transpose(-2, -3).transpose(-1, -2))

        cnnFusion = torch.concat(features[0:-1], dim=2)
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

    def getKernelSize(self, dates: int) -> list:
        thrsh = (dates - 3) // 2
        result = [1]
        while result[-1] < thrsh:
            result.append(result[-1] * 5)
        del result[-1]
        return result


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
                in_channels=self.embeddingSize[0] * 2,
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


# Parallelized Convolutional-Recurrent Network
class PCRN(nn.Module):
    def __init__(
        self,
        dates,
        ninputs,
        noutputs,
        hiddens,
        nlayers,
        fusions,
        embeddings: (int, int),
    ):
        super().__init__()
        self.encoder = Encoder(dates, ninputs, hiddens, nlayers, fusions, embeddings)
        self.decoder = Decoder(dates, noutputs, embeddings)

    def forward(self, x):
        result = self.encoder(x)
        # result = self.decoder(result)
        return result


def Config2PCRN(path, device) -> (PCRN, int):
    with open(path) as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        pcrn = yml["pcrn"]

    model = PCRN(
        pcrn["term"],
        len(pcrn["inputs"]),
        len(pcrn["outputs"]),
        pcrn["hiddens"],
        pcrn["nlayers"],
        pcrn["fusions"],
        tuple(tuple(map(int, pcrn["embeddings"].split(",")))),
    )

    if "weigth" in pcrn:
        model.load_state_dict(torch.load(pcrn["weight"], map_location=device))

    model.to(device)

    return model, pcrn["term"], pcrn["inputs"]
