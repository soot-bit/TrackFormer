
import math
from torch import softmax, nn, optim
import torch
import numpy as np
from torch.nn.functional import mse_loss, l1_loss
import lightning as L
from abc import ABC, abstractmethod





################################### Helper functions:



def scaled_dot_product(q, k, v, mask=None):
    """
    Performs scaled dot-product attention.
    
    Args:
        q, k, v (torch.Tensor)  : Query , Key , value  tensors  (B, num_heads, seq_len, head_dim).
        mask : batch firts mask
    """

    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor

    attention = torch.softmax(attn_weight, dim=-1)
    values = attention @ v
    return values, attention





#################################### TrackFormer layers:

class MultiheadAttention(nn.Module):
    
    """
    Multihead attention mechanism.
    ------------------------------

    Args:
        input_dim (int): Input dimension.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must divisible among heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # d_k

        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim) #stacked matrices
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

        # Separate Q, K, V 
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [B, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [B, SeqLen, Head, Dims]
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
    """
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):

        super().__init__()

        # Attention 
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # ff
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers Norms
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





class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, trainer):
        self.warmup = warmup
        if hasattr(trainer.train_dataloader, '__len__'):
            self.max_num_iters = trainer.max_epochs * len(trainer.train_dataloader)
        else:
            self.max_num_iters = trainer.max_epochs * trainer.limit_train_batches
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor




class Loss:
    def __init__(self, mode='mse'):
        super().__init__()
        self.mode = mode
        self.quantile = None
        
        if "qloss" in self.mode:
            _, q = self.mode.split("-")
            self.quantile = float(q)
            self.loss_fn = self._quantile_loss
        elif 'mse' in self.mode:
            self.loss_fn = mse_loss
        elif 'mae' in self.mode:  
            self.loss_fn = l1_loss
        else:
            raise ValueError(f"Uknown loss funtion: {self.mode}")

    def _quantile_loss(self, preds, targets):
        errors = targets - preds
        return torch.mean(torch.max((self.quantile - 1) * errors, self.quantile * errors))

    def __call__(self, preds, targets):
        return self.loss_fn(preds, targets)


class BaseModel(L.LightningModule):
    '''
    Base LightningModule for training and evaluating models.
    Optimiser args:
        lr (float): Learning rate.
        warmup (int): Number of warmup steps, [50, 500].
        max_iters (int): Maximum number of iterations the model is trained for, used by the CosineWarmup scheduler.
    Loss args:
        loss_type (dic or str): Type of loss function 'mse', "mae' or 'qloss- q_value'.
        quantile (float, optional): Quantile value for quantile loss, if used. Default is 0.5.
    '''

    def __init__(self, criterion, lr, warmup):
        super().__init__()
        self.criterion = Loss(criterion)
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             trainer=self.trainer)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    
    def _calculate_loss(self, batch, mode="train"):

        inputs, _, label = batch

        preds = self(inputs)
        loss = self.criterion(preds.squeeze(), label.squeeze())
        self.log(f"{mode}_loss", loss, prog_bar=True, logger=True, batch_size=inputs.shape[0] )
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")
    
    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
