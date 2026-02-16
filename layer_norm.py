import torch
import torch.nn as nn

## nlp和cv中的layer_norm略有不同
## nlp example
batch, seq_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, seq_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
final_result = layer_norm(embedding)

## cv example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)


