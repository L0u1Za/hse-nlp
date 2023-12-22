from torch import nn
import torch

class EmbedFactorized(nn.Module):
    def __init__(self, vocab_size, H, E=128, padding_idx=None, embedding_matrix=None):
        super().__init__()

        self.emb1 = nn.Embedding(vocab_size, E, padding_idx)
        self.emb2 = nn.Embedding(E, H, padding_idx)

        if embedding_matrix is not None:
            u, s, vt = torch.svd(embedding_matrix)
            u = u[:, :E]
            s = torch.diag(s[:E])
            vt = vt[:E,:]
            vt = torch.matmul(s, vt)
            self.emb1.weight.data = u.contiguous()
            self.emb2.weight.data = vt.contiguous()

    def forward(self, inputs):
        output = torch.bmm(self.emb1(inputs), self.emb2.weight.data.unsqueeze(0).repeat(inputs.shape[0], 1, 1))
        return output