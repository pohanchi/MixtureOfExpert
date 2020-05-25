import torch
import torch.nn as nn
import numpy as np
from kmeans_pytorch import kmeans





class RoutingAttention(nn.Module):
    def __init__(self, heads):
        super().__init__()

        self.n_heads = heads
    
    def forward(self, qk):
        qk = qk.detach().to(torch.float16)
        batch_size, seq_len, dim, device = *qk.shape, qk.device # [6,12,384,64]

        W_R = torch.empty(batch_size, dim, dim, device=device)
        nn.init.orthogonal_(W_R)
        
        W_R = W_R.to(torch.float16)
        R = torch.matmul(qk, W_R).reshape(-1, dim)
        K = int(seq_len ** 0.5)
        
        import pdb
        pdb.set_trace()
        cluster_idx, centroid = kmeans(X=R, num_clusters=K, distance='cosine', device=device)
        
        cluster_idx = cluster_idx.reshape(batch_size, seq_len).unsqueeze(1).expand(-1, self.n_heads, -1)
        result = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=device)
        # import pdb 
        # pdb.set_trace()
        r1 = result.to(torch.long) + cluster_idx.unsqueeze(-1).to(device)   # [0, 0, 0, 0, ...]
        r2 = result.to(torch.long) + cluster_idx.unsqueeze(-2).to(device)  # [0, 1, 2, 3, ...]

        result = (r1 == r2).to(torch.float32)
        result = 10000. * result - 10000. 
        
        return result.detach()
        
