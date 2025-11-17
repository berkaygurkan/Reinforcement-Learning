
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class RL2RecurrentPolicy(nn.Module):
    """GRU tabanlı RL² policy + value ağı.
    Girdi: x_t (ör. [obs, a_{t-1}, r_{t-1}, d_{t-1}])
    Çıkış: action, log_prob, value, next_hidden
    """
    def __init__(self, input_dim:int, action_dim:int, hidden_size:int=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim  = action_dim

        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.gru   = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.fc_pi = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.fc_v  = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size:int=1, device:str="cpu"):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward_step(self, x_t:torch.Tensor, h:torch.Tensor):
        """Tek zaman adımı için ileri geçiş.
        x_t: (B, input_dim), h: (1, B, H)
        """
        z = F.relu(self.fc_in(x_t))
        z = z.unsqueeze(1)              # (B,1,H)
        out, h_next = self.gru(z, h)    # out: (B,1,H)
        out = out.squeeze(1)            # (B,H)

        mean = self.fc_pi(out)
        std  = self.log_std.exp().unsqueeze(0)
        dist = Normal(mean, std)
        action = dist.sample()
        logp   = dist.log_prob(action).sum(dim=-1)
        value  = self.fc_v(out).squeeze(-1)
        return action, logp, value, h_next

    def act_deterministic(self, x_t:torch.Tensor, h:torch.Tensor):
        z = F.relu(self.fc_in(x_t))
        z = z.unsqueeze(1)
        out, h_next = self.gru(z, h)
        out = out.squeeze(1)
        mean = self.fc_pi(out)
        value = self.fc_v(out).squeeze(-1)
        return mean, value, h_next
