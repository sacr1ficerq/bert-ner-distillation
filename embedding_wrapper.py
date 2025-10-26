import torch.nn as nn
import torch
class EmbeddingWrapper(nn.Module):
    def __init__(self, big_embeddings: nn.Embedding, rank: int):
        super().__init__()
        vocab_size = big_embeddings.num_embeddings
        output_size = big_embeddings.embedding_dim

        UT, S, V = torch.linalg.svd(big_embeddings.weight, full_matrices=False)
        UT = UT.to('cuda:0')
        V = V.to('cuda:0')
        s = torch.sqrt(S[:rank]).to('cuda:0')
        A = UT[:, :rank] * s
        B = torch.diag(s) @ V[:rank, :]

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=rank, padding_idx=big_embeddings.padding_idx)
        self.embedding.weight.data = A

        self.projection = nn.Linear(in_features=rank, out_features=output_size, bias=False)
        self.projection.weight.data = B.T

    def forward(self, x):
        x = self.embedding(x)
        x = self.projection(x)
        return x