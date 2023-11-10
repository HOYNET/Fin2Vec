import torch
from torch_geometric.data import Data

# 가정: 4개의 노드와 3개의 엣지가 있음
# 엣지 인덱스: 0 -> 1, 1 -> 2, 2 -> 3
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

# 엣지 가중치: 각 엣지에 대한 가중치
edge_weight = torch.tensor([0.5, 0.7, 0.9], dtype=torch.float)

# PyG 데이터 객체 생성
data = Data(edge_index=edge_index, edge_attr=edge_weight)

print(data.is_directed())


nodes = torch.tensor([[2, 1, 3], [2, 3, 4], [3, 4, 5]], dtype=torch.float)
edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
edge_weights = torch.tensor([0.5, 0.7], dtype=torch.float)

data = Data(x=nodes, edge_index=edges, edge_attr=edge_weights)
print(data.is_directed())
