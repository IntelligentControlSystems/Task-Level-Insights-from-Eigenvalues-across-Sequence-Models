''' Copyright (c) 2025 ETH Zurich, Institute for Dynamics Systems and Control, Rahel Rickenbach, 
Jelena Trisovic, Alexandre Didier, Jerome Sieber, Melanie N. Zeilinger. No rights reserved. '''

import torch
from torch import nn
import torch.nn.functional as F
import math
from jax import random
import jax.numpy as np
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import eigh

# reference implementation of the matching layer used in the LRA retrieval task
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/layers/common_layers.py#L197
class MATCH(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, mlp_dim)
        self.middle = nn.Linear(mlp_dim, int(mlp_dim//2))
        self.decoder = nn.Linear(int(mlp_dim//2), output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.middle(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x

# reference implementation of the transformer MLP
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/layers/common_layers.py#L144
class MLP(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim=None, dropout=0.0):
        super().__init__()
        self.output_dim = input_dim if output_dim is None else output_dim
        self.encoder = nn.Linear(input_dim, mlp_dim)
        self.decoder = nn.Linear(mlp_dim, self.output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.decoder(x)
        x = self.dropout(x)
        return x

class GLU(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = self.linear(x)
        out = out[:, :, :x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2]:])
        return self.dropout(out)

class LAMBDA(nn.Module):
    def __init__(self, input_dim, init=0.5, dropout=0.0):
        super().__init__()
        self.dim = input_dim
        self.encoder = nn.Linear(input_dim, input_dim * 2)
        self.decoder = nn.Linear(input_dim * 2, input_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        init = -math.log(1/init - 1)
        self.alpha = nn.Parameter(torch.ones((1,))*init)

    def forward(self, x):
        xz = self.encoder(x)
        a = F.sigmoid(self.alpha)
        out = a*self.glu(xz) + (1-a)*self.mlp(xz)
        return self.dropout(out)
    
    def glu(self, xz):
        x, z = torch.split(xz, [self.dim, self.dim], dim=-1)
        return x * torch.sigmoid(z)
    
    def mlp(self, x):
        x = self.activation(x)
        x = self.dropout(x)
        return self.decoder(x)

# reference implementation of the classifier head implemented in LRA
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/layers/common_layers.py#L166
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_classes, pooling):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.pooling = pooling
        if self.mlp_dim != 0:
            self.encoder = nn.Linear(input_dim, mlp_dim)
            self.decoder = nn.Linear(mlp_dim, num_classes)
            self.activation = nn.ReLU()
        
    def forward(self, x):
        # pooling
        if self.pooling in ["mean"]:
            x = torch.mean(x, dim=1)
        elif self.pooling in ["max"]:
            x = torch.max(x, dim=1)[0]
        elif self.pooling in ["sum"]:
            x = torch.sum(x, dim=1)
        elif self.pooling in ["cls"]: # if classifier scalar is learnt concurrently
            x = x[:,0,:]
        else:
            x = x # no pooling
        
        if self.mlp_dim != 0:
            x = self.encoder(x)
            x = self.activation(x)
            x = self.decoder(x)
        return x

class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        learnable: bool = True,
        device='cuda',
        dtype='torch.float32',
    ):
        """
        GPT-2 Learnable Token and Position Embeddings.
        If max_position_embeddings <= 0, there's no position embeddings
        We embed to word_embe_proj_dim dimension then project up to embed_dim
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                word_embed_proj_dim,
                padding_idx=padding_idx,
            )
            self.project_in = nn.Linear(
                word_embed_proj_dim, embed_dim, bias=False
            )
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim
            )
    
    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape[:2]
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=self.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

### SSM Initialization methods

def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(key, shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return np.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)
