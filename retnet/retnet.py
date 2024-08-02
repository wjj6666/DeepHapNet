import torch
from torch import nn
from retnet.retention import MultiScaleRetention

class RetNet(nn.Module):
    def __init__(self,hidden_dim: int=512):
        super(RetNet, self).__init__()
        ffn_size = 4*hidden_dim
        heads = 4
        layers = 3
       

        self.layers = layers    # 模型的层数
        self.hidden_dim = hidden_dim    # 隐藏层维度
        self.ffn_size = ffn_size       # Feed-Forward Network（FFN）的大小
        self.heads = heads             # 注意力头的数量
        # self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim          # 如果使用了双倍的v_dim，则设置v_dim为隐藏维度的两倍

        # 初始化模型中的每一层的多尺度保留机制、FFN和LayerNorm层
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads)    # 每一层使用一个多尺度保留机制
            for _ in range(layers)      # 根据层数进行循环
        ]) 
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),      # 线性变换层，将隐藏层维度映射到FFN的大小
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)       # 线性变换层，将FFN的输出映射回隐藏层维度
            )
            for _ in range(layers)     # 根据层数进行循环
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)     # 第一个LayerNorm层，用于输入数据前
            for _ in range(layers)       # 根据层数进行循环
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)     # 第二个LayerNorm层，用于FFN的输出后
            for _ in range(layers)       # 根据层数进行循环
        ])
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):     # 根据层数进行循环
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X     # 使用多尺度保留机制对输入进行处理，然后与原始输入相加
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y      # 使用FFN对处理后的输入进行处理，然后与上一步的结果相加
        return X, torch.matmul(X, X.transpose(1,2))
