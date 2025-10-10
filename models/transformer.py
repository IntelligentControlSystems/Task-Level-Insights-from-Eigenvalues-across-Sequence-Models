import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

from models import MATCH, MLP, GLU, LAMBDA, ClassifierHead, TokenEmbeddings
from models import MHA, MHNA

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from flash_attn import flash_attn_qkvpacked_func
if torch.__version__.split("+")[0] != "2.3.1":
    from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
else:
    print("[IMPORTANT] Flash linear attention is not supported for this installation!")


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, cfg, layer_idx=None):
        super().__init__()
        # remove configs only used at this level
        d_model = hidden_dim
        d_qk = cfg["state_dim"]
        num_heads = cfg["num_heads"]
        att_dropout = cfg["att_dropout"]
        mixer = cfg["mixer"]
        mixer_dim = cfg["mixer_dim"]
        dropout = cfg["dropout"]
        norm = cfg["norm"]
        use_flash = cfg["use_flash"] if "use_flash" in cfg else False
        use_gate = cfg["use_gate"] if "use_gate" in cfg else False
        conv_type = cfg["conv_type"] if "conv_type" in cfg else "full"
        self.attention_fn = cfg["attention_fn"]

        # attention function
        if self.attention_fn in ["sm-attention"]:
            dim_conv = cfg["dim_conv"]  if "dim_conv" in cfg else 0
            self.attention = MHA(d_model, d_qk, num_heads, dim_conv=dim_conv,
                                 lin_att=False, dropout=att_dropout, use_flash=use_flash, conv_type=conv_type)
        elif self.attention_fn in ["lin-attention"]:
            dim_conv = cfg["dim_conv"]  if "dim_conv" in cfg else 0
            self.attention = MHA(d_model, d_qk, num_heads, dim_conv=dim_conv,
                                 lin_att=True, dropout=att_dropout, use_flash=use_flash, conv_type=conv_type)
        elif self.attention_fn in ["norm-attention"]:
            mode = cfg["mode"]
            norm_fn = cfg["norm_fn"]
            approx_fn = cfg["approx_fn"]
            scale_B = cfg["scale_B"]
            offset = cfg["offset"]
            offset_init = cfg["offset_init"]
            learn_A = cfg["learn_A"]
            dim_conv = cfg["dim_conv"]
            self.attention = MHNA(d_model, d_qk, num_heads, mode, norm_fn, approx_fn,
                                  scale_B, offset, offset_init, learn_A, dim_conv,
                                  dropout=att_dropout, use_flash=use_flash, layer_idx=layer_idx, conv_type=conv_type)

        # use gate branch or not
        self.use_gate = use_gate
        if self.use_gate:
            self.Wz = nn.Linear(d_model, d_model)
            nn.init.constant_(self.Wz.bias, 1.0)
            nn.init.xavier_uniform_(self.Wz.weight, gain=0.1)
        
        # MLP/GLU
        if mixer in ["mlp"]:
            self.mixer = MLP(hidden_dim, mixer_dim, dropout=dropout)
            self.drop_skip = False
        elif mixer in ["glu"]:
            self.mixer = GLU(hidden_dim)
            self.drop_skip = False
        elif mixer in ["hybrid"]:
            self.mixer = LAMBDA(hidden_dim, init=0.2, dropout=dropout)
            self.drop_skip = False
        elif mixer in ["none"]:
            self.mixer = nn.Identity(hidden_dim)
            self.drop_skip = True
        else:
            raise RuntimeError("{0} mixer not implemented yet!".format(mixer))

        if norm in ["layer"]:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise RuntimeError("{0} norm not implemented yet!".format(norm))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_gate:
            z = self.Wz(x)
        skip = x
        x = self.norm(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + skip

        y = self.norm(x)
        y = self.mixer(y)

        if self.drop_skip:
            if self.use_gate:
                y = y * F.silu(z)
        else:
            if self.use_gate:
                y = (x + y) * F.silu(z)
            else:
                y = x + y

        return y #if self.drop_skip else x + y
    
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()      
        # remove configs only used at this level
        input_dim = cfg["input_dim"]
        output_dim = cfg["output_dim"]
        num_layers = cfg["num_layers"]
        hidden_dim = cfg["hidden_dim"]
        embed = cfg["embedding"]
        vocab_size = cfg["vocab_size"]
        max_len = cfg["max_pos_embed"]
        pooling = cfg["pooling"]
        self.dual = cfg["dual"]
        self.classify = cfg["classifier"]

        mixer_dim = cfg["mixer_dim"]
        norm = cfg["norm"]
        dropout = cfg["dropout"]

        if embed:
            self.encoder = TokenEmbeddings(hidden_dim, vocab_size, max_len)
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.Sequential(*[TransformerBlock(hidden_dim, cfg, idx) for idx in range(num_layers)])
        if self.classify:
            self.classifier = ClassifierHead(hidden_dim, mixer_dim, output_dim, pooling)
        else:
            self.decoder = nn.Linear(hidden_dim, output_dim, bias=False)
        if self.dual:
            self.match = MATCH(output_dim*2, mixer_dim, output_dim)
        if norm in ["layer"]:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise RuntimeError("{0} norm not implemented yet!".format(norm))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.norm(x)
        if self.classify:
            x = self.classifier(x)
            if self.dual:
                (x1, x2) = torch.split(x, int(x.shape[0]/2))
                x = self.match(torch.concatenate((x1, x2), dim=1))
        else:
            x = self.decoder(x)
        return x
