import torch
from torch import nn

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_states, num_heads):
        super().__init__()
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.head_dim = hidden_states // num_heads

        self.q_linear = nn.Linear(hidden_states,hidden_states)
        self.k_linear = nn.Linear(hidden_states,hidden_states)
        self.v_linear = nn.Linear(hidden_states,hidden_states)

        self.o_linear = nn.Linear(hidden_states,hidden_states)

    def split_head(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None):
        bacth_size = hidden_states.size()[0]

        q = self.q_linear(hidden_states)
        k = self.k_linear(hidden_states)
        v = self.v_linear(hidden_states)

        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        attention_score = torch.matmul(q, k.transpose(-1,-2)) / torch.sqrt(torch.tensor(self.head_dim))

        if attention_mask != None:
            attention_score += (1 - attention_mask) * -1e-9

        attention_map = torch.softmax(attention_score, dim=-1)

        output = torch.matmul(attention_map, v)

        output = output.transpose(1, 2).contiguous().view(bacth_size, -1 ,self.num_heads * self.head_dim)

        output = self.o_linear(output)

        return output