import torch
import torch.nn as nn
import math

class InputEmbedding():
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ## create a matrix of shape seq_len X d_model
        pe = torch.zeros(seq_len, d_model)

        ## create a vector of shape seq_len X 1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        ## define the div_term - which is the log of an exponential to help keep things mathematically stable
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        ## apply a sin conversion to the even places and a cosine conversion to the odd places
        pe[:, 0::2] = torch.sin(div_term * position)
        pe[:, 1:2] = torch.cos(div_term * position)

        # convert to shape of (1, seq_len, d_model) to apply it to the batch of sentences
        pe = pe.unsqueeze(0)

        ## register on buffer
        self.register('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False) # because we don't want the model to learn the positional encoding and it will always be constant
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10e-6) -> None:
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # additive parameter
        self.bias = nn.Parameter(torch.zeros(1)) # multiplicative parameter

    def forwards(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
     
