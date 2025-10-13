''' Copyright (c) 2025 ETH Zurich, Institute for Dynamics Systems and Control, Rahel Rickenbach, 
Jelena Trisovic, Alexandre Didier, Jerome Sieber, Melanie N. Zeilinger. No rights reserved. '''

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from flash_attn import flash_attn_qkvpacked_func

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v):
        """Implements multihead softmax attention.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.dropout(attention)
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        return output

class FlashAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout_p = dropout

    def forward(self, qkv):
        """Implements flash inner attention.
        Arguments
        ---------
            qkv: Tensor containing the queries, keys and values. (B, S, 3, H, D)
        """
        softmax_scale = 1.0 / math.sqrt(qkv.shape[-1])
        output = flash_attn_qkvpacked_func(
            qkv.to(torch.float16), # need to convert to float16 for flash attention
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=softmax_scale,
            causal=True
        )
        return output.to(torch.float32)


class SelfLinAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v):
        """Implements multihead linear attention.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        kv = torch.einsum("bshd,bsht->bshdt",k,v)
        kv = torch.cumsum(kv, dim=1)

        k = torch.cumsum(k, dim=1)
        n = torch.einsum("bshd,bshd -> bsh", q, k)
        n = n.pow(-1)

        output = torch.einsum("bshd,bshdt->bsht",q,kv)
        output = n[:,:,:,None]*output
        return self.dropout(output)
    
class MHA(nn.Module):
    """Multi-head self-attention
    """
    def __init__(
        self,
        d_model: int,
        d_qk: int=None,
        num_heads: int=1,
        dim_conv: int=0,
        lin_att: bool=True,
        dropout: float=0.0,
        bias: bool=True,
        use_flash: bool=True,
        layer_idx: int=None,
        conv_type: str="full" # full or partial
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.conv_type = conv_type
        if d_qk is None:
            self.d_qk = d_model
        else:
            self.d_qk = d_qk

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_qk % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        assert (
            self.d_model % num_heads == 0
        ), "self.vdim must be divisible by num_heads"
        self.head_dim = self.d_qk // num_heads
        self.v_dim = self.d_model // num_heads
        self.use_flash = use_flash
        self.Wqkv = nn.Linear(
            d_model, 2 * self.d_qk + d_model, bias=bias
        )

        if lin_att:
            self.inner_attn = SelfLinAttention(dropout)
        else:
            if self.use_flash and self.head_dim == self.v_dim: # use flash attention
                self.inner_attn = FlashAttention(dropout)
            else: # use naive attention
                self.inner_attn = SelfAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        self.use_conv = True if dim_conv > 0 else False
        if dim_conv > 0:
            self.act = nn.SiLU()
            if conv_type == "full":
                conv_dim = self.d_model + 2 * self.d_qk
            else:
                conv_dim = 2 * self.d_qk
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=True,
                kernel_size=dim_conv,
                groups=conv_dim,
                padding=dim_conv - 1,
            )

    def forward(self, x: torch.Tensor):
        """"""
        qkv = self.Wqkv(x)
        if self.use_conv:
            seqlen = qkv.shape[1]
            if self.conv_type == "full":
                qkv = self.act(self.conv1d(qkv.transpose(1, 2))[..., :seqlen])
                qkv = qkv.transpose(1, 2)
            else:
                qk, v = torch.split(qkv, [2 * self.d_qk, self.d_model], dim=-1)
                qk = self.act(self.conv1d(qk.transpose(1, 2))[..., :seqlen])
                qk = qk.transpose(1, 2)
                qkv = torch.cat([qk, v], dim=-1)

        if self.use_flash and self.head_dim == self.v_dim: # use flash attention
            qkv = rearrange(
                qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
            )
            context = self.inner_attn(qkv)
            out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
            return out
        else: # use naive attention
            qk, v = torch.split(
                qkv, [2 * self.d_qk, self.d_model], dim=-1
            )
            qk = rearrange(
                qk, "... (two h d) -> ... two h d", two=2, d=self.head_dim
            )
            v = rearrange(
                v, "... (h d) -> ... h d", d=self.v_dim
            )
            context = self.inner_attn(qk, v)
            out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
            return out
