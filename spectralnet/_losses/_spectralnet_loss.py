import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(
        self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D)[:, None]


        # 余弦相似度
        # Y_normalized = F.normalize(Y, p=2, dim=1) # Normalize each vector in Y
        # S = torch.mm(Y_normalized, Y_normalized.t()) # Compute cosine similarity matrix
        # Dy = 1 - S # Convert similarity to distance

        # 欧氏距离
        Dy = torch.cdist(Y, Y)

        # 曼哈顿距离
        #Dy = torch.cdist(Y, Y, p=1)  # Use p=1 for Manhattan distance

        loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        return loss
