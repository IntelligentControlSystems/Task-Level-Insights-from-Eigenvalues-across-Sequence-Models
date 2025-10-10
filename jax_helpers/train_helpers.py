import os
import shutil
from functools import partial
import jax
import jax.numpy as jnp
from jax.nn import one_hot
from jax.nn import log_softmax
from tqdm import tqdm
from flax.training import train_state, orbax_utils
import optax
import orbax.checkpoint as ocp
from typing import Any


### LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = jnp.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def constant_lr(step, base_lr, end_step, lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, lr_min = lr_params # could add custom wd or lr for S5 or S4 (not yet)

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states["regular"].inner_state.hyperparams["learning_rate"] = jnp.array(
        lr_val, dtype=jnp.float32
    )
    state.opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"] = jnp.array(
        ssm_lr_val, dtype=jnp.float32
    )
    return state, step


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc

###

### create train state

def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {k: (map_fn(v) if hasattr(v, "keys") else fn(k, v)) for k, v in nested_dict.items()}

    return map_fn


def create_train_state_s5(model_cls, rng, in_dim, batch_size, seq_len, weight_decay, norm, ssm_lr, ssm_vars, lr, padded, betas):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param in_dim:
    :param batch_size:
    :param seq_len:
    :param weight_decay:
    :param norm:
    :param ssm_lr:
    :param lr:
    :param padded:
    :return:

    MISSING:
    :param opt_config:
    :param retrieval:
    """

    if padded:
        dummy_input = (jnp.ones((batch_size, seq_len, in_dim)), jnp.ones(batch_size))
    else:
        dummy_input = jnp.ones((batch_size, seq_len, in_dim))
    
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)
    
    if norm in ["batch"]:
        params = variables["params"]
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"]

    encoder_params = params.get("encoder", {}).get("encoder", {})

    ## TODO: make this more general or specific to the SSM model used
    # Smaller lr and no weight decay for lambda, gamma and B
    ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
    tx = optax.multi_transform(
        {
            "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
            "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                weight_decay=weight_decay),
        },
        ssm_fn,
    )
    ##

    fn_is_complex = lambda x: x.dtype in [jnp.complex64, jnp.complex128]

    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    nr_params = sum(jax.tree_util.tree_leaves(param_sizes))

    encoder_param_sizes = map_nested_fn(
    lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(encoder_params)
    encoder_nr_params = sum(jax.tree_util.tree_leaves(encoder_param_sizes))

    if norm in ["batch"]:
        class TrainState(train_state.TrainState):
            batch_stats: Any

        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats), (nr_params, encoder_nr_params)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), (nr_params, encoder_nr_params)



def create_train_state(model_cls, rng, in_dim, batch_size, seq_len, weight_decay, norm, ssm_lr, ssm_vars, lr, padded, betas):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param in_dim:
    :param batch_size:
    :param seq_len:
    :param weight_decay:
    :param norm:
    :param ssm_lr:
    :param lr:
    :param padded:
    :return:

    MISSING:
    :param opt_config:
    :param retrieval:
    """

    if padded:
        dummy_input = (jnp.ones((batch_size, seq_len, in_dim)), jnp.ones(batch_size))
    else:
        dummy_input = jnp.ones((batch_size, seq_len, in_dim))
    
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)
    
    if norm in ["batch"]:
        params = variables["params"]
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"]

    encoder_params = params.get("encoder", {}).get("encoder", {})

    ## TODO: make this more general or specific to the SSM model used
    # Smaller lr and no weight decay for lambda, gamma and B
    ssm_fn = map_nested_fn(
        lambda k, _: "ssm"
        if k in ssm_vars else "regular"
    )
    tx = optax.multi_transform(
        {
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr, b1=betas[0], b2 = betas[1]),
            "regular": optax.inject_hyperparams(optax.adamw)(
                learning_rate=lr, weight_decay=weight_decay, b1=betas[0], b2 = betas[1]
            ),
        },
        ssm_fn,
    )
    ##

    fn_is_complex = lambda x: x.dtype in [jnp.complex64, jnp.complex128]

    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    nr_params = sum(jax.tree_util.tree_leaves(param_sizes))

    encoder_param_sizes = map_nested_fn(
    lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(encoder_params)
    encoder_nr_params = sum(jax.tree_util.tree_leaves(encoder_param_sizes))

    if norm in ["batch"]:
        class TrainState(train_state.TrainState):
            batch_stats: Any

        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats), (nr_params, encoder_nr_params)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), (nr_params, encoder_nr_params)

###

### Train and eval steps
@jax.vmap
def batched_average_mask(a, mask):
    """Average of a by sum of values of mask"""
    return a / jnp.sum(mask)


@jax.vmap
def create_mask(x, length):
    L = x.shape[0]
    mask = (jnp.arange(L) >= length[0]) * (jnp.arange(L) < length[1])
    return mask


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * jax.nn.log_softmax(logits, axis=-1))


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits) == label


def compute_accuracies(logits, labels):
    if len(logits.shape) == 4:
        return jnp.sum(
            compute_accuracy(logits, labels).mean(axis=-1),
            axis=-1,
        )
    elif len(logits.shape) == 2 or len(logits.shape) == 3:
        return jnp.mean(compute_accuracy(logits, labels))


def loss_fn(logits, labels):
    """
    Pick the desired loss depending on the shape of the logits (and therefore the task)
    """
    if len(logits.shape) == 2 or len(logits.shape) == 3:  # for classification tasks
        losses = cross_entropy_loss(logits, labels)
    if len(logits.shape) == 4:  # for tasks with multidimensional dense targets
        losses = cross_entropy_loss(logits, labels).mean(axis=-1)
    return jnp.mean(losses)


def prep_batch(batch, seq_len, in_dim, lang_model=False):
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """

    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Unhandled data type. ")

    inputs = jnp.array(inputs.numpy()).astype(float)  # convert to jax
    targets = jnp.array(targets.numpy())  # convert to jax
    lengths = aux_data.get("lengths", None)  # get lengths from aux if it is there.

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        inputs = jnp.pad(inputs, ((0, 0), (0, num_pad)), "constant", constant_values=(0,))

    # Inputs size is [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(inputs, in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None and not lang_model:
        lengths = jnp.array(lengths)
        full_inputs = (inputs.astype(float), lengths.astype(float))
    else:
        full_inputs = inputs.astype(float)

    return full_inputs, targets


@partial(jax.jit, static_argnums=(4, 5))
def train_step(state, rng, inputs, labels, model, norm):
    """Performs a single training step given a batch of data"""

    def _loss(params):
        if norm in ["batch"]:
            p = {"params": params, "batch_stats": state.batch_stats}
            logits, vars = model.apply(p, inputs, rngs={"dropout": rng}, mutable=["batch_stats"])
        else:
            p = {"params": params}
            vars = None
            logits = model.apply(p, inputs, rngs={"dropout": rng})
        return loss_fn(logits, labels), vars

    (loss, vars), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)

    if norm in ["batch"]:
        state = state.apply_gradients(grads=grads, batch_stats=vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, norm, lr_params):
    """
    Training function for an epoch that loops over batches.
    """
    model = model(training=True)  # model in training mode
    batch_losses = []
    decay_function, ssm_lr, lr, step, end_step, lr_min = lr_params

    for batch in tqdm(trainloader):
        inputs, labels = prep_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(state, drop_rng, inputs, labels, model, norm)
        batch_losses.append(loss)  # log loss value

        lr_params = (decay_function, ssm_lr, lr, step, end_step, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, jnp.mean(jnp.array(batch_losses)), step


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(inputs, labels, state, model, norm):
    if norm == "batch":
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats}, inputs)
    else:
        logits = model.apply({"params": state.params}, inputs)
    losses = loss_fn(logits, labels)
    accs = compute_accuracies(logits, labels)
    return jnp.mean(losses), accs, logits


def validate(state, model, testloader, seq_len, in_dim, norm, lang_model=False):
    """Validation function that loops over batches"""
    model = model(training=False)
    losses, accuracies = jnp.array([]), jnp.array([])

    for batch in tqdm(testloader):
        inputs, labels = prep_batch(batch, seq_len, in_dim, lang_model)
        loss, acc, logits = eval_step(inputs, labels, state, model, norm)
        losses = jnp.append(losses, loss)
        accuracies = jnp.append(accuracies, acc)
    return jnp.mean(losses), jnp.mean(accuracies)

###

### save jax model

def save_model(path, ckpt):
    if os.path.isabs(path):
        ckpt_dir = path
    else:
        ckpt_dir = os.path.abspath(path.strip("/"))
    
    print("-----\nSaving model to {0}\n-----".format(ckpt_dir))

    # Remove the existing directory if it exists
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.save(ckpt_dir, ckpt, save_args=save_args)

###