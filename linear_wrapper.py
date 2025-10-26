import torch
import torch.nn as nn
from typing import Union


class LinearWrapper(nn.Module):
    def __init__(self, big_linear: nn.Linear, rank: Union[int, float]):
        in_features, out_features = big_linear.weight.shape

        super().__init__()

        W = big_linear.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        if isinstance(rank, float):
            if not (0.0 < rank < 1.0):
                raise ValueError("rank not in valid range [0, 1) and not an int")

            total_variance = torch.sum(S**2)
            cumulative_variance = torch.cumsum(S**2, dim=0)
            explained_variance_ratio = cumulative_variance / total_variance

            rank_to_use = (torch.searchsorted(explained_variance_ratio, rank, right=True)).item()

        elif isinstance(rank, int):
            if not (0 < rank <= S.size(0)):
                raise ValueError(f"Rank {rank} is out of valid range [1, {S.size(0)}]")
            rank_to_use = rank

        s = torch.sqrt(S[:rank_to_use])
        A = U[:, :rank_to_use] * s
        B = torch.diag(s) @ Vh[:rank_to_use, :]

        self.linear_A = nn.Linear(in_features=in_features, out_features=rank_to_use, bias=False)
        self.linear_A.weight.data = A.T
        
        self.linear_B = nn.Linear(in_features=rank_to_use, out_features=out_features, bias=(big_linear.bias is not None))
        self.linear_B.weight.data = B.T

        if big_linear.bias is not None:
            self.linear_B.bias.data = big_linear.bias.data

        self.rank = rank_to_use
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_A(x)
        x = self.linear_B(x)
        return x
