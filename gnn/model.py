import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class GNN(nn.Module):
    def __init__(self, ninput, noutput, nlayer, dropout=0.5):
        super(GNN, self).__init__()
        self.ninput, self.noutput, self.dropout, self.nlayer, self.nhidden = (
            ninput,
            noutput,
            dropout,
            nlayer,
            16,
        )

        self.convs = nn.ModuleList([gnn.GCNConv(self.ninput, self.nhidden)])
        for i in range(self.nlayer - 1):
            self.convs.append(gnn.GCNConv(self.nhidden, self.nhidden * 2))
            self.nhidden *= 2

        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(self.nhidden, self.noutput)

    def forward(self, x, edgeIndex, edgeWeight):
        for i, m in enumerate(self.consConvs):
            x = self.relu(m(x, edgeIndex, edge_weight=edgeWeight))
            x = self.dropout(x)

        result = self.dense(x)
        return result


class VTXGNN(nn.Module):
    def __init__(self, ninput, noutput, nlayer, dropout=0.5):
        super(VTXGNN, self).__init__()
        self.ninput, self.noutput, self.nlayer, self.dropout = (
            ninput,
            noutput,
            nlayer,
            dropout,
        )
        self.splyGNN = GNN(self.ninput, self.noutput, self.nlayer, self.dropout)
        self.consGNN = GNN(self.ninput, self.noutput, self.nlayer, self.dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(2 * noutput, self.noutput)

    def forward(self, x, splyEdgeIndex, splyEdgeWeight, consEdgeIndex, consEdgeWeight):
        sply = self.splyGNN(x, splyEdgeIndex, splyEdgeWeight)
        cons = self.consGNN(x, consEdgeIndex, consEdgeWeight)

        result = self.relu(torch.cat([sply, cons], dim=-1))
        result = self.dense(result)
        return result, sply, cons
