import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.head_dim)

        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def spilt_head(self, x, num_query = None):
        batch_size = x.size()[0]
        if num_query == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            return x.view(batch_size, -1, num_query, self.head_dim).transpose(1, 2)
        
    def forward(self, hidden_size, mask_attention = None):
        batch_size = hidden_size.size()[0]

        q = self.q_linear(hidden_size)
        k = self.k_linear(hidden_size)
        v = self.v_linear(hidden_size)

        q = self.spilt_head(q)
        k = self.spilt_head(k, 1)
        v = self.spilt_head(v, 1)

        attention_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))

        if mask_attention != None:
            attention_score += (1 - mask_attention) * -1e-9

        attention_map = torch.softmax(attention_score, dim=-1)

        output = torch.matmul(attention_map, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1 , self.head_dim * self.num_heads)

        output = self.o_linear(output)

        return output



