import torch
import torch.nn as nn

batch_size, seq_len, dim = 2, 4, 8
x = torch.randn(batch_size, seq_len, dim)

layer_norm = nn.LayerNorm(dim, elementwise_affine=True)
ln_output = layer_norm(x)

instance_norm = nn.InstanceNorm2d(seq_len, affine=True)
## 因为 nn.InstanceNorm2d（或类似 instance_norm）通常要求输入是 4D：[N, C, H, W]
in_output = instance_norm(x.reshape(batch_size, seq_len, dim, 1)).reshape(batch_size, seq_len, dim)

print("LayerNorm Output:\n", ln_output)
print("InstanceNorm Output:\n", in_output)
print("Outputs Close:", torch.allclose(ln_output, in_output))

print("TP in layer_norm:", sum(p.numel() for p in layer_norm.parameters() if p.requires_grad))
print("TP in instance_norm:", sum(p.numel() for p in instance_norm.parameters() if p.requires_grad))

