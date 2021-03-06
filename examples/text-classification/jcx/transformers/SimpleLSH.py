import torch
import torch.nn as nn


class SimpleLSHAttention16(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, qk_, attention_mask_, bucket_size=32, **kwargs):
        qk = qk_.detach().to(torch.float16)
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device # [6,12,384,64]
        attention_mask = attention_mask_.to(torch.float16)

        M = torch.max(torch.norm(qk, dim=-1))
        qk_norm = qk / M
        # qk_norm = qk.div(torch.norm(qk, dim=-1, keepdim=True))
        qk_const = torch.norm(qk_norm, dim=-1, keepdim=True)

        qk_const = torch.sqrt(1. - torch.pow(qk_const, 2))
        qk = torch.cat((qk_norm, qk_const), -1)
        a = torch.randn([batch_size, n_heads, seq_len, dim+1], device=device).to(torch.float16).normal_(mean=0, std=1)
        Q = torch.sum(qk.mul(a), dim=-1)           # [6, 12, 384]
        
        # a_P = torch.unsqueeze(a, -2)
        # a_P = a_P.expand(-1, -1, -1, seq_len, -1) # [6, 12, 384, 384, 64+2]
        # a_P = a_P.permute(2, 0, 1, 3, 4)          # [384, 6, 12, 384, 64+2]
        # P = torch.sum(a_P.mul(qk), dim=-1)         # [384, 6, 12, 384]

        P = torch.matmul(qk, a.transpose(-1, -2)).permute(3, 0, 1, 2)
        
        result = Q.unsqueeze(0).mul(P)                             # [384, 6, 12, 384]
        result = result.permute(1, 2, 0, 3)
        # Deal with Nan
        result[result!=result] = 0.
        result = result + attention_mask
        try:
            max_idx = torch.topk(result, k=bucket_size, dim=-1)[1]
        except:
            import pdb
            pdb.set_trace()
        result = torch.ones(result.shape, device=device) * (-10000.)
        result.scatter_(-1, max_idx, 0.)
        """
        result_0 = torch.zeros(result.shape, device=device)
        result_10000 = torch.ones(result.shape, device=device) * (-10000.)
        result = torch.where(result>0, result_0, result_10000)
        """
        return result.detach()

class SimpleLSHAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, qk, bucket_size=32, **kwargs):
        qk = qk.detach()
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device # [6,12,384,64]

        qk_norm = qk.div(torch.norm(qk, dim=-1, keepdim=True))
        qk_const = torch.norm(qk_norm, dim=-1, keepdim=True)

        qk_const = torch.sqrt(1. - torch.pow(qk_const, 2))
        qk = torch.cat((qk, qk_const), -1)
        a = torch.randn([batch_size, n_heads, seq_len, dim+1], device=device).normal_(mean=0, std=1)
        Q = torch.sum(qk.mul(a), dim=-1)           # [6, 12, 384]
        
        a_P = torch.unsqueeze(a, -2)
        a_P = a_P.expand(-1, -1, -1, seq_len, -1) # [6, 12, 384, 384, 64+2]
        a_P = a_P.permute(2, 0, 1, 3, 4)          # [384, 6, 12, 384, 64+2]

        P = torch.sum(a_P.mul(qk), dim=-1)         # [384, 6, 12, 384]
        
        result = Q.unsqueeze(0).mul(P)                             # [384, 6, 12, 384]
        result = result.permute(1, 2, 0, 3)
        # Deal with Nan
        result[result!=result] = 0.

        max_idx = torch.topk(result, k=bucket_size, dim=-1)[1]
        result = torch.ones(result.shape, device=device) * (-10000.)
        result.scatter_(-1, max_idx, 0.)
        """
        result_0 = torch.zeros(result.shape, device=device)
        result_10000 = torch.ones(result.shape, device=device) * (-10000.)
        result = torch.where(result>0, result_0, result_10000)
        """
        
        
        return result.detach()


