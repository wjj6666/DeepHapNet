"""
定义了两个类:
MultiScaleRetention类和SimpleRetention类
SimpleRetention 类
一、SimpleRetention 类原理：
SimpleRetention 类实现了简单的保留机制，主要是根据输入的矩阵进行一系列的矩阵运算，然后得到保留矩阵，以控制信息在模型中的保留与丢弃。

二、SimpleRetention 类作用和功能：
SimpleRetention 类主要用于在输入数据中保留相关的信息，通过对输入数据的处理，学习数据之间的关系，并且根据这些关系生成保留矩阵，以便后续的模型使用。
MultiScaleRetention 类

三、MultiScaleRetention 类原理：
MultiScaleRetention 类实现了多尺度的保留机制，它在每个时间步或者每个数据块上，采用不同尺度的简单保留机制进行处理，然后将每个尺度的输出合并起来。

四、MultiScaleRetention 类作用和功能：
MultiScaleRetention 类的作用是结合多个不同尺度的保留机制，以更好地捕获输入数据的不同层次的信息，从而提高模型的表达能力和性能。它能够处理输入数据中不同时间步或不同块的信息，并且通过不同尺度的保留机制来对这些信息进行处理，以便后续的模型使用。

五、区别：
单一 vs. 多尺度：
SimpleRetention 类只实现了单一尺度的保留机制，而 MultiScaleRetention 类则实现了多个不同尺度的保留机制。

功能复杂度：
MultiScaleRetention 类功能更为复杂，因为它可以处理更多层次的信息，并且可以结合多个尺度的信息来提高模型的性能。

适用范围：
SimpleRetention 类更适合简单的保留机制任务，而 MultiScaleRetention 类则更适合处理复杂的数据结构，例如序列数据或者图像数据等。

六、三种前向传播方式
1、并行表示Parallel Representation:
这种传播方式是将输入数据同时传递给每个保留机制，每个机制对数据进行处理并产生输出，然后将这些输出合并成最终的结果。
适用于同时处理数据的场景，例如处理多个输入的情况。

2、循环表示Recurrent Representation:
在这种传播方式中，输入数据按时间步序列依次传递给每个保留机制，每个机制根据当前输入和前一时间步的保留状态产生输出，并更新保留状态，然后将所有输出合并成最终结果。
适用于需要考虑时间序列信息的场景，例如处理序列数据的情况。

3、分块表示Chunkwise Representation:
这种传播方式将输入数据分成多个块，然后每个块依次传递给每个保留机制，机制根据当前块的输入和前一块的保留状态产生输出，并更新保留状态，然后将所有输出合并成最终结果。
适用于处理较大数据集时，为了减少计算量而将数据分成多个块处理的情况。

"""
import math

import torch
import torch.nn as nn

from retnet.xpos_relative_position import XPOS

"""
定义了一个简单的保留机制模型SimpleRetention,其中包含了初始化函数__init__。
该模型根据指定的参数初始化权重矩阵W_Q、W_K和W_V,并引入了XPOS模块用于处理相对位置编码。
"""
class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        
        self.xpos = XPOS(head_size)


    """
    实现了保留机制的前向传播函数。
    该函数接受输入X,并通过计算权重矩阵W_Q和W_K的乘积,以及相对位置编码,生成保留矩阵ret,再与W_V的乘积得到最终输出。
    """
    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(self.W_Q.device)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)

        K = self.xpos(K, downscale=True)

        V = X @ self.W_V

        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        return ret @ V

    """
    实现了保留机制的循环表示。
    该函数接受输入x_n和上一个时间步的保留状态n_s_1,并根据当前时间步n,计算当前时间步的保留状态n_s。
    """
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)
        
        return (Q @ s_n), s_n
    

    """
    实现了保留机制的分块表示。
    该函数接受输入x_i和上一个时间步的保留状态r_i_1,并根据当前块索引i,计算当前时间步的保留状态r_i。
    """
    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V
        
        r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V
        
        #e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)
        
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)
        
        cross_chunk = (Q @ r_i_1) * e
        
        return inner_chunk + cross_chunk, r_i

    """
    实现了一个内部函数 _get_D(sequence_length)，用于生成一个矩阵 D,该矩阵用于控制保留机制中的相对权重
    """
    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)   # 创建一个列向量 n，其元素从 0 到 sequence_length-1，每个元素都表示一个位置
        m = torch.arange(sequence_length).unsqueeze(0)   # 创建一个行向量 m，与 n 同理。

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m   计算 n - m 的幂次方，即将每个位置的相对距离指数化。然后通过 n >= m 创建一个掩码矩阵，其中位置 n < m 的元素为 0，其他元素为 1。最后，将这两个矩阵相乘，得到最终的相对权重矩阵 D。
        # fill the NaN with 0
        D[D != D] = 0   # 将 D 中的 NaN 值（因为 gamma ** (n - m) 的结果可能导致无穷大或零除以零的情况）替换为 0，以确保数值的稳定性和合理性。

        return D
    

"""
定义了一个多尺度保留机制的模型MultiScaleRetention,
其中包含了初始化函数__init__。在初始化函数中,
定义了模型的各种参数,
包括隐藏大小、头数、是否双倍v_dim等。
同时计算了一系列不同尺度的gamma值,并创建了多个简单保留机制的实例。
"""
class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()

        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    """
    实现了多尺度保留机制的前向传播函数forward。该函数接受输入X
    并将其送入每个简单保留机制中，并将每个保留机制的输出连接起来，并进行了归一化处理。
    然后，通过使用 Swish 激活函数计算结果并返回。
    """
    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
    

    """
    实现了多尺度保留机制的循环表示forward_recurrent。
    该函数接受输入x_n和上一个时间步的保留状态s_n_1s,并将每个简单保留机制应用于x_n的一个片段,
    并将每个保留机制的输出连接起来，并进行了归一化处理。然后，通过使用 Swish 激活函数计算结果并返回。
    """
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """
    
        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns


    """
    实现了多尺度保留机制的分块表示forward_chunkwise。
    该函数接受输入x_i和上一个时间步的保留状态r_i_1s,
    并将每个简单保留机制应用于x_i的一个片段,
    并将每个保留机制的输出连接起来，并进行了归一化处理。
    然后，通过使用 Swish 激活函数计算结果并返回。
    """
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
                )
            Y.append(y)
            r_is.append(r_i)
        
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is
