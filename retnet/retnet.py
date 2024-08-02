import torch
from torch import nn
from retnet.retention import MultiScaleRetention

class RetNet(nn.Module):
    def __init__(self,hidden_dim: int=512):
        super(RetNet, self).__init__()
        ffn_size = 4*hidden_dim
        heads = 4
        layers = 3
       

        self.layers = layers    
        self.hidden_dim = hidden_dim   
        self.ffn_size = ffn_size       
        self.heads = heads             
        # self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim         


        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads)   
            for _ in range(layers)      
        ]) 
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),      
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)      
            )
            for _ in range(layers)     
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)    
            for _ in range(layers)       
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)    
            for _ in range(layers)      
        ])
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):    
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X     
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y      
        return X, torch.matmul(X, X.transpose(1,2))
