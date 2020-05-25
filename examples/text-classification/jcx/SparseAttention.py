import torch
import torch.nn as nn
import math


class SparseAttentionStrided(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, qk, attention_mask, bucket_size):
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device
        l = math.floor(seq_len ** 0.5)
        local_mask = torch.ones(seq_len, seq_len, device=device)
        local_mask = torch.triu(local_mask, -l).permute(1, 0)
        local_mask = torch.triu(local_mask, -l)
        local_mask = torch.where(local_mask==1, torch.tensor(0., device=device), torch.tensor(-10000., device=device))
        local_mask = torch.unsqueeze(torch.unsqueeze(local_mask, 0), 0).expand(batch_size, int(n_heads/2), -1, -1)

        
        x = torch.arange(seq_len, device=device).unsqueeze(-1).to(torch.float32)
        y = x.permute(1, 0)
        z = torch.zeros(seq_len, seq_len, device=device)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = (torch.fmod(q-k, l) == 0)
        # global_mask = (c1 * c2).to(torch.float32)
        global_mask = c2.to(torch.float32)
        global_mask = torch.where(global_mask==1, torch.tensor(0., device=device), torch.tensor(-10000., device=device))
        global_mask = torch.unsqueeze(torch.unsqueeze(global_mask, 0), 0).expand(batch_size, int(n_heads/2), -1, -1)

        result = torch.cat((local_mask, global_mask), 1)

        return result.detach()


        

        
        


