import torch
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_head, group):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.head_dim = hidden_size // num_head
        self.group = group
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, group * self.head_dim)
        self.v_linear = nn.Linear(hidden_size, group * self.head_dim)

        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def splti_head(self, x, group=None):
        batch_size, seq_len = x.size()[:2]
        if group == None:
            return x.view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        else:
            x = x.view(batch_size, -1, group, self.head_dim).transpose(1, 2)
            x = x[:,:,None,:,:].expand(batch_size, group, self.num_head // group, seq_len, self.head_dim).reshape(batch_size, self.num_head // group * group, seq_len, self.head_dim)
            return x

    def forward(self, hidden_size, attention_mask=None):
        batch_size = hidden_size.size()[0]

        q = self.q_linear(hidden_size)
        k = self.k_linear(hidden_size)
        v = self.v_linear(hidden_size)

        q = self.splti_head(q)
        k = self.splti_head(k, self.group)
        v = self.splti_head(v, self.group)

        attention_score = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))

        if attention_mask != None:
            attention_score += (1 - attention_mask) * -1e-9

        attention_map = torch.softmax(attention_score, dim=-1)

        output = torch.matmul(attention_map, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_head)

        output = self.o_linear(output)

        return output   