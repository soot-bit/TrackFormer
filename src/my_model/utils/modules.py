
import math
from torch import softmax, nn, optim
import torch
import numpy as np
from torch.nn.functional import mse_loss







################################### Helper functions:



def scaled_dot_product(q, k, v, mask=None):
    """
    Performs scaled dot-product attention.
    
    Args:
        q (torch.Tensor): Query tensor  (B, num_heads, seq_len, head_dim).
        k (torch.Tensor): Key tensor  (B, num_heads, seq_len, head_dim).
        v (torch.Tensor): Value tensor (B, num_heads, seq_len, head_dim).
        mask (torch.Tensor, optional): ?
    """

    scale_factor = 1 / math.sqrt(q.size(-1)) 
    attn_weight = q @ k.transpose(-2, -1) * scale_factor

    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)  # [Batch, 1, 1, SeqLen]
        attn_weight = attn_weight.masked_fill(mask == 0, float("-inf"))

    attention = softmax(attn_weight, dim=-1)
    values = attention @ v
    return values, attention





class LossFunction:
    def __init__(self, loss_function: str = "qloss", quantile: float = 0.5):
        self.quantile = quantile
        if loss_function == "qloss":
            self.loss_fn = self.quantile_loss
        elif loss_function == "mse":
            self.loss_fn = mse_loss
        else:
            raise ValueError("Invalid loss function. Choose either 'quantile' or 'mse'.")

    def __call__(self, preds, targets):
        return self.loss_fn(preds, targets)

    def quantile_loss(self, preds, targets):
        errors = targets - preds
        return torch.mean(torch.max((self.quantile - 1) * errors, self.quantile * errors))




#################################### TrackFormer layers:

class MultiheadAttention(nn.Module):
    
    """
    Multihead attention mechanism.
    ------------------------------

    Args:
        input_dim (int): Input dimension.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.

    Attributes:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        qkv_proj (nn.Linear): Linear layer to project the input into query, key, and value tensors.
        o_proj (nn.Linear): Linear layer to project the output of the attention mechanism.
    """

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must divisible among heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # d_k

        # Stacked weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    
    """
    Transformer encoder block.

    Args:
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Attributes:
        self_attn (MultiheadAttention): Multihead attention layer.
        linear_net (nn.Sequential): Feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after the attention layer.
        norm2 (nn.LayerNorm): Layer normalization after the feedforward network.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):

        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # ff
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention 
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FeedForward
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x



class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """Positional Encoding.

        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # [SeqLen, HiddenDim] 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)


        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x



class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
