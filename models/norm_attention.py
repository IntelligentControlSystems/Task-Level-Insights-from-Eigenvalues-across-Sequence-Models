import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
if torch.__version__.split("+")[0] != "2.3.1":
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
else:
    print("[IMPORTANT] Flash linear attention is not supported for this installation!")

def init_offset(size, a=0.02, b=0.1, lo=8., hi=14.):
    if size == 1:
        return torch.tensor([(hi-lo)/2])
    else:
        x = torch.log(torch.expm1(torch.linspace(a, b, size)))
        x = (x - x.min()) / (x.max() - x.min())
        x = x * abs(hi-lo) + lo
        return x


class SelfNormAttention(nn.Module):
    def __init__(self, norm_fn, approx_fn, num_heads=1, scale_B=False, offset=False, offset_init="uniform", dropout=0.0):
        super().__init__()
        if norm_fn in ["exp"]:
            self.norm_fn = lambda x: torch.exp(x)
        elif norm_fn in ["elu"]:
            self.norm_fn = lambda x: F.elu(x)
        elif norm_fn in ["softplus"]:
            self.norm_fn = lambda x: F.softplus(x)
        elif norm_fn in ["sigmoid"]:
            self.norm_fn = lambda x: F.sigmoid(x)
        else:
            raise RuntimeError("normalization function {0} not implemented!".format(norm_fn))
        
        if approx_fn in ["none"]:
            self.approx_fn = lambda x: x
        elif approx_fn in ["elu"]:
            self.approx_fn = lambda x: F.elu(x) + 1
        else:
            raise RuntimeError("approximation function {0} not implemented!".format(approx_fn))
        
        if offset:
            if offset_init in ["uniform"]:
                self.offset = nn.Parameter(init_offset(num_heads))
            elif offset_init in ["exp"]:
                self.offset = nn.Parameter(torch.linspace(4, 9, num_heads))
            else:
                raise RuntimeError("Invalid init option {0}".format(offset_init))
        else:
            self.offset = None
        
        self.scale_B = scale_B
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, n):
        """Implements the multihead linear attention with normalization.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
            n:  Tensor containing the normalization values. (B, S, H)
        """
        q, k = qk.unbind(dim=2)
        q = self.approx_fn(q)
        k = self.approx_fn(k)

        if self.scale_B:
            scale = 1.0 / math.sqrt(q.shape[-1])
        else:
            scale = 1.0

        kv = torch.einsum("bshd,bsht->bshdt",k * scale,v)
        kv = torch.cumsum(kv, dim=1)

        output = torch.einsum("bshd,bshdt->bsht",q,kv)

        if self.offset is not None:
            n = torch.exp(-self.norm_fn(n + self.offset))
        else:
            n = torch.exp(-self.norm_fn(n))
        output = n[:,:,:,None]*output

        return self.dropout(output)

class FlashNormAttention(nn.Module):
    def __init__(self, norm_fn, approx_fn, num_heads=1, scale_B=False, offset=False, offset_init="uniform", dropout=0.0):
        super().__init__()
        if norm_fn in ["exp"]:
            self.norm_fn = lambda x: torch.exp(x)
        elif norm_fn in ["elu"]:
            self.norm_fn = lambda x: F.elu(x)
        elif norm_fn in ["softplus"]:
            self.norm_fn = lambda x: F.softplus(x)
        elif norm_fn in ["sigmoid"]:
            self.norm_fn = lambda x: F.sigmoid(x)
        else:
            raise RuntimeError("normalization function {0} not implemented!".format(norm_fn))
        
        if approx_fn in ["none"]:
            self.approx_fn = lambda x: x
        elif approx_fn in ["elu"]:
            self.approx_fn = lambda x: F.elu(x) + 1
        else:
            raise RuntimeError("approximation function {0} not implemented!".format(approx_fn))
        
        if offset:
            if offset_init in ["uniform"]:
                self.offset = nn.Parameter(init_offset(num_heads))
            elif offset_init in ["exp"]:
                self.offset = nn.Parameter(torch.linspace(4, 9, num_heads))
            else:
                raise RuntimeError("Invalid init option {0}".format(offset_init))
        else:
            self.offset = None
        
        self.scale_B = scale_B
        self.dropout = nn.Dropout(dropout)

        self.mode = "chunk"

    def forward(self, qk, v, n):
        """Implements the multihead linear attention with normalization.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
            n:  Tensor containing the normalization values. (B, S, H)
        """
        q, k = qk.unbind(dim=2)
        q = self.approx_fn(q)
        k = self.approx_fn(k)

        if self.scale_B:
            scale = 1.0 / math.sqrt(q.shape[-1])
        else:
            scale = 1.0

        if self.mode == "chunk":
            output, _ = chunk_simple_gla(q, k, v, None, scale)
        elif self.mode == "fused":
            output, _ = fused_recurrent_simple_gla(q, k, v, None, scale)
        else:
            raise RuntimeError("Invalid mode option {0}".format(self.mode))

        if self.offset is not None:
            n = torch.exp(-self.norm_fn(n + self.offset))
        else:
            n = torch.exp(-self.norm_fn(n))
        output = n[:,:,:,None]*output

        return self.dropout(output)


class MHNA(nn.Module):
    """Multi-head self-attention with normalization
    """
    def __init__(
        self,
        d_model: int,
        d_qk: int=None,
        num_heads: int=1,
        mode: str="attention",
        norm_fn: str="exp",
        approx_fn: str="none",
        scale_B: bool=False,
        offset: bool=False,
        offset_init: str="uniform",
        learn_A: bool=False,
        dim_conv: int=0,
        dropout: float=0.0,
        use_flash: bool=False,
        bias: bool=True,
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

        self.Wvqkn = nn.Linear(
            d_model, d_model + 2 * self.d_qk + num_heads, bias=bias
        )

        if mode in ["attention"]:
            if use_flash:
                self.inner_attn = FlashNormAttention(norm_fn, approx_fn, num_heads, scale_B, offset, offset_init, dropout)
            else:
                self.inner_attn = SelfNormAttention(norm_fn, approx_fn, num_heads, scale_B, offset, offset_init, dropout)
        else:
            raise RuntimeError("Invalid mode option {0}".format(mode))
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
        seqlen = x.shape[1]
        vqkn = self.Wvqkn(x)
        vqk, n = torch.split(
            vqkn, [self.d_model + 2 * self.d_qk, self.num_heads], dim=-1
        )
        if self.use_conv:
            if self.conv_type == "full":
                vqk = self.act(self.conv1d(vqk.transpose(1, 2))[..., :seqlen])
                vqk = vqk.transpose(1, 2)
            else:
                v, qk = torch.split(vqk, [self.d_model, 2 * self.d_qk], dim=-1)
                qk = self.act(self.conv1d(qk.transpose(1, 2))[..., :seqlen])
                qk = qk.transpose(1, 2)
                vqk = torch.cat([v, qk], dim=-1)

            
        v, qk = torch.split(
            vqk, [self.d_model, 2 * self.d_qk], dim=-1
        )
        qk = rearrange(
            qk, "... (two h d) -> ... two h d", two=2, d=self.head_dim
        )
        v = rearrange(
            v, "... (h d) -> ... h d", d=self.v_dim
        )
        context = self.inner_attn(qk, v, n)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    