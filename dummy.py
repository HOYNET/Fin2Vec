import torch
from torch import nn
from hoynet import Hoynet

batches, features, seqs = 5, 7, 80
input = torch.rand(batches, features, seqs)

encoder = nn.Linear(seqs, 5)
decoder = None

model = Hoynet(encoder, 5, 10, 5, 5, 5, decoder)
print(model(input).shape)
