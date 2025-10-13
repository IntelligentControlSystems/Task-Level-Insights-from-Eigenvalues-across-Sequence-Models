''' Copyright (c) 2025 ETH Zurich, Institute for Dynamics Systems and Control, Rahel Rickenbach, 
Jelena Trisovic, Alexandre Didier, Jerome Sieber, Melanie N. Zeilinger. No rights reserved. '''

from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn


class SequenceLayer(nn.Module):
    """ Defines a single SSM layer, with custom SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (e.g. S5, LRU)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
            dropout     (float32):  dropout rate
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            norm        (string):   norm to apply, either batch or layer
    """
    ssm: nn.Module
    d_model: int
    dropout: float = 0.0
    activation: str = "full_glu"
    training: bool = True
    prenorm: bool = True
    norm: str = "layer"

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm()

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        if self.norm in ["batch"]:
            self.normalize = nn.BatchNorm(use_running_average=not self.training, axis_name='batch')
        else:
            self.normalize = nn.LayerNorm()

        self.drop = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not self.training,
        )

    def __call__(self, x):
        """
        Compute the LxH output of SSM layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm:
            x = self.normalize(x)
        x = self.seq(x)

        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.normalize(x)
        return x

# TODO: could also add embedding from annotated S4!
class StackedEncoderModel(nn.Module):
    """ Defines a stack of SSM layers to be used as an encoder.
        Args:
            ssm         (nn.Module): the SSM to be used (e.g. S5, LRU)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                     we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            norm        (string):   norm to apply, either batch or layer
    """
    ssm: nn.Module
    d_model: int
    n_layers: int
    activation: str = "full_glu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = True
    norm: str = "layer"

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                ssm=self.ssm,
                dropout=self.dropout,
                d_model=self.d_model,
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                norm=self.norm,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x


def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = jnp.arange(L) < lengths
    return jnp.sum(mask[..., None]*x, axis=0)/lengths


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


class ClassificationModel(nn.Module):
    """ SSM classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of SSM layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (e.g. S5, LRU)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            pooling     (str):      Options: [mean: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            norm        (string):   norm to apply, either batch or layer
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool = False
    activation: str = "full_glu"
    dropout: float = 0.2
    training: bool = True
    pooling: str = "mean"
    prenorm: bool = True
    norm: str = "layer"

    def setup(self):
        """
        Initializes the stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            norm=self.norm,
                    )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x)
        if self.pooling in ["mean"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)
        elif self.pooling in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]
        elif self.pooling in ["none"]:
            x = x # do not pool at all
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last', 'none']")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "batch_stats": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)

## TODO: Add Retrieval models for document matching task (AAN)
