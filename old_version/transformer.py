import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, nb_heads):
        super().__init__()

        self.nb_heads = nb_heads
        self.d_k = d_k
        self.d_v = d_v 

        #all heads together for efficiency
        self.query = nn.Linear(d_model, d_k * nb_heads)
        self.key = nn.Linear(d_model, d_k * nb_heads)
        self.value = nn.Linear(d_model, d_v * nb_heads)

        self.fc = nn.Linear(d_v * nb_heads, d_model)
     

    def forward(self, q, k, v, mask=None):
        q = self.query(q)   # batch_size x seq_length x (d_k * nb_heads)
        k = self.key(k)     # batch_size x seq_length x (d_k * nb_heads)
        v = self.value(v)   # batch_size x seq_length x (d_v * nb_heads)

        batch_size = q.shape[0] 
        seq_length = q.shape[1]

        #separate the heads and reorder
        # batch_size x seq_length x nb_heads x d -> batch_size x nb_heads x seq_length x d
        q = q.view(batch_size, seq_length, self.nb_heads, self.d_k).transpose(1,2)
        k = k.view(batch_size, seq_length, self.nb_heads, self.d_k).transpose(1,2)
        v = v.view(batch_size, seq_length, self.nb_heads, self.d_v).transpose(1,2)

        # now I understand how tensor multiplication works :))) The last two dimensions are subjected to normal matmul
        # (batch_size x nb_heads x seq_length x d_k)(batch_size x nb_heads x d_k x seq_length) -> (batch_size x nb_heads x seq_length x seq_length)
        attention = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention = attention.masked_fill(mask[:, None, None, :] == 0, float('-inf')) #None fills in the gaps in the shape with ones.

        # apply softmax along the last dimension    
        attention = F.softmax(attention, dim = -1)

        # apply attention to the values
        # (batch_size x nb_heads x seq_length x seq_length)(batch_size x nb_heads x seq_length x d_v) -> (batch_size x nb_heads x seq_length x d_v) 
        attention = attention @ v

        # need (batch_size x seq_length x (nb_heads * d_v))
        attention = attention.transpose(1,2).contiguous().view(batch_size, seq_length, self.nb_heads * self.d_v)

        return self.fc(attention)
    


#use the MultiHeadAttention to implement a Transformer block
class TransformerBlock(nn.Module):
    """max_len=None => encoder model"""
    def __init__(self, d_k, d_v, d_model, nb_heads, max_len=None, d_ff=None, dropout_proba=0.1):
        super().__init__()
        if max_len is not None:
            self.mha = CausalSelfAttention(d_k, d_v, d_model, nb_heads, max_len)
        else:
            self.mha = MultiHeadAttention(d_k, d_v, d_model, nb_heads)
        self.dropout1 = nn.Dropout(dropout_proba)
        if d_ff is None:
            d_ff = d_model * 4 #default...seems like this is a popular choice?
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  #or ReLU or...?
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_proba),
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attention_out = self.mha(x, x, x, mask)
        #dropout option
        drop1 = self.dropout1(attention_out)
        add_norm1 = self.layer_norm1(x + drop1)
        ff_out = self.ff(add_norm1)
        add_norm2 = self.layer_norm2(add_norm1 + ff_out)
        #option to dropout here too
        return add_norm2



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_proba=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_proba)
        position = torch.arange(max_len).unsqueeze(1)
        # equivalent to 10000 ^ (-2*i/d_model), which will be like dividing by 10000 ^ (2*i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.) / d_model)) #someone at pytorch thinks this exp and -log business is safer!
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # batch_size x seq_length x embedding_dim
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class CausalSelfAttention(MultiHeadAttention):
    def __init__(self, d_k, d_v, d_model, nb_heads, max_len):
        super().__init__(d_k, d_v, d_model, nb_heads)
        #lower triangular of ones
        causal_mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('causal_mask', causal_mask.view(1, 1, max_len, max_len))

    def forward(self, q, k, v, pad_mask=None):
        q = self.query(q)   # batch_size x seq_length x (d_k * nb_heads)
        k = self.key(k)     # batch_size x seq_length x (d_k * nb_heads)
        v = self.value(v)   # batch_size x seq_length x (d_v * nb_heads)

        batch_size = q.shape[0] 
        seq_length = q.shape[1]

        #separate the heads and reorder
        # batch_size x seq_length x nb_heads x d -> batch_size x nb_heads x seq_length x d
        q = q.view(batch_size, seq_length, self.nb_heads, self.d_k).transpose(1,2)
        k = k.view(batch_size, seq_length, self.nb_heads, self.d_k).transpose(1,2)
        v = v.view(batch_size, seq_length, self.nb_heads, self.d_v).transpose(1,2)

        # (batch_size x nb_heads x seq_length x d_k)(batch_size x nb_heads x d_k x seq_length) -> (batch_size x nb_heads x seq_length x seq_length)
        attention = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        
        if pad_mask is not None:
            attention = attention.masked_fill(
                pad_mask[:, None, None, :] == 0, float('-inf')
            )
        attention = attention.masked_fill(
            self.causal_mask[:, :, :seq_length, :seq_length] == 0, float('-inf')
        )
        attention = F.softmax(attention, dim = -1)

        # apply attention to the values
        # (batch_size x nb_heads x seq_length x seq_length)(batch_size x nb_heads x seq_length x d_v) -> (batch_size x nb_heads x seq_length x d_v) 
        attention = attention @ v

        # need (batch_size x seq_length x (nb_heads * d_v))
        attention = attention.transpose(1,2).contiguous().view(batch_size, seq_length, self.nb_heads * self.d_v)

        return self.fc(attention)

    
class Decoder(nn.Module):
    def __init__(self, 
                    d_k, 
                    d_v, 
                    d_model, 
                    nb_heads, 
                    nb_layers, 
                    dropout_proba, 
                    max_len, 
                    vocab_size,
                    d_ff=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout_proba, max_len)
        blocks = [
            TransformerBlock(d_k, d_v, d_model, nb_heads, max_len, d_ff=d_ff, dropout_proba=dropout_proba)
            for _ in range(nb_layers)
        ]
        self.transformer_blocks = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask=None):
        x = self.embedding(x)
        x = self.pe(x)
        if pad_mask is not None:
            for block in self.transformer_blocks:
                x = block(x, pad_mask)
        else:
            x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x

