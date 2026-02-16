import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zero(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
features = 768
layer_norm = LayerNorm(features)

x = torch.randn(10, 20, features)
output = layer_norm(x)
print(output)