import torch
import torch.nn.functional as F
import einops
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sys import path
from os import getcwd
import jax
from jax import random
import jax.numpy as jnp
from jax.numpy.linalg import eigh, inv, matrix_power
from functools import partial
import orbax.checkpoint as ocp
from causal_conv1d import causal_conv1d_fn

import wandb
import os
import tempfile

from models import Mamba, Transformer
from models import BatchClassificationModel, init_S4, init_S5, init_LRU


def get_layers(layer_type, model):
    # load input an pass it through layers
    if layer_type == "mamba":
        nr_layers = len(model.blocks)
        layers = []
        for i in range(nr_layers):
            layers.append(model.blocks[i])
    elif layer_type == "transformer":
        nr_layers = len(model.layers)
        layers = []
        for i in range(nr_layers):
            layers.append(model.layers[i])

    return layers

def get_eig_att_softmax(x, layer, d_qk, num_heads, d_model):
    # obtain keys and querries

    qkv = layer.attention.Wqkv(x)
    qk, v = torch.split(qkv, [2 * d_qk, d_model], dim=-1)
    qk = einops.rearrange(qk, "... (two h d) -> ... two h d", two=2, d=layer.attention.head_dim)

    # assign q and k accordingly
    q = qk[:,:,0,:,:]
    k = qk[:,:,1,:,:]

    seqlen = qk.shape[1]

    # calculate scores
    scores = torch.einsum("bthd,bshd->btsh", q, k)
    mask_mul = torch.tril(torch.full((seqlen, seqlen), 1, device=scores.device), 0)
    scores = torch.einsum("btsh,ts->btsh", scores, mask_mul.to(dtype=scores.dtype))

    # make calculation numerical feasible by subtracting largest row score
    scores_max = torch.max(scores,-2).values

    # repeat to get correct dimensions
    scores_max_r = einops.repeat(scores_max, 'a i j ->a i newaxis j', newaxis=seqlen)

    # create mask to get lower triangular matrix
    mask_mul = torch.tril(torch.full((seqlen, seqlen), 1, device=scores.device), 0)
    scores_max_r = torch.einsum("btsh,ts->btsh", scores_max_r, mask_mul.to(dtype=scores.dtype))

    # calculate row normalized score
    scores_norm = scores - scores_max_r
    scores_norm = scores_norm.detach().cpu().numpy()
    scores_norm = scores_norm.astype(np.float64)

    # get elementwise exponential (row-wise normalized)
    exp_scores = np.nan_to_num(np.exp(scores_norm))

    # get nu (row-wise normalized)
    nu = exp_scores.sum(axis=2)
 
    # get eigenvalues by dividing nu=i with nu=i+1
    eta = np.divide(nu[:,:-1,:],nu[:,1:,:])

    # division of two values with different scaling/normalization requires multiplication with inverse scaling/normalization
    scores_max_np = scores_max.detach().cpu().numpy()
    score_max_diff = -scores_max_np[:,1:,:]+scores_max_np[:,:-1,:]
    max_scaling = np.exp(score_max_diff.astype(np.float64))

    eta = eta*max_scaling

    # add dimension for concatenation 
    eta = np.expand_dims(eta, axis=-1)

    return eta

def get_eig_att_linear(x, layer, d_qk, num_heads, d_model):
    # obtain keys and querries
    qkv = layer.attention.Wqkv(x)
    qk, v = torch.split(
                qkv, [2 * d_qk, d_model], dim=-1
            )
    qk = einops.rearrange(qk, "... (two h d) -> ... two h d", two=2, d=layer.attention.head_dim)


    # assign q and k accordingly
    q = qk[:,:,0,:,:]
    k = qk[:,:,1,:,:]
    q = torch.nn.functional.elu(q) + torch.ones(q.shape, device='cuda')
    k = torch.nn.functional.elu(k) + torch.ones(q.shape, device='cuda')

    seqlen = qk.shape[1]

    # calculate scores
    scores = torch.einsum("bthd,bshd->btsh", q, k)
    mask_mul = torch.tril(torch.full((seqlen, seqlen), 1, device=scores.device), 0)
    scores = torch.einsum("btsh,ts->btsh", scores, mask_mul.to(dtype=scores.dtype))

    scores = scores.detach().cpu().numpy()
    scores = scores.astype(np.float64)

    # get elementwise exponential (row-wise normalized)
    scores = np.nan_to_num(scores)

    # get nu (row-wise normalized)
    nu = scores.sum(axis=2)
    nu[nu == 0.0] = 2e-23 #(ok ?)

    # get eigenvalues by dividing nu=i with nu=i+1
    eta = np.divide(nu[:,:-1,:],nu[:,1:,:])
    
    # add dimension for concatenation 
    eta = np.expand_dims(eta, axis=-1)
    
    return eta

def get_eig_att_norm(x, layer, d_qk, num_heads, d_model, model_config):
    norm_fn_cf = model_config['norm_fn']
    approx_fn_cf = model_config['approx_fn']
    offset = model_config['offset']

    if norm_fn_cf in ["exp"]:
        norm_fn = lambda x: torch.exp(x)
    elif norm_fn_cf in ["elu"]:
        norm_fn = lambda x: torch.nn.functional.elu(x)
    elif norm_fn_cf in ["softplus"]:
        norm_fn = lambda x: torch.nn.functional.softplus(x)
    elif norm_fn_cf in ["sigmoid"]:
        norm_fn = lambda x: torch.nn.functional.sigmoid(x)
    else:
        raise RuntimeError("normalization function {0} not implemented!".format(norm_fn_cf))
    
    # obtain keys and querries
    vqkn = layer.attention.Wvqkn(x)

    vqk, n = torch.split(
            vqkn, [d_model + 2 * d_qk, num_heads], dim=-1
        )

    if offset:
        n = torch.exp(-norm_fn(n + layer.attention.inner_attn.offset))
    else:
        n = torch.exp(-norm_fn(n))

    n = n.detach().cpu().numpy()
    n = n.astype(np.float64)
    n[n == 0.0] = 2e-23 #(ok ?)

    eta = np.divide(n[:,1:,:],n[:,:-1,:])

    # add dimension for concatenation 
    eta = np.expand_dims(eta, axis=-1)

    return eta

def get_eig_mamba2(x, layer):
    xbcdt = layer.mamba.in_proj(x)  # (B, L, d_in_proj)
    A = -torch.exp(layer.mamba.A_log)  # (nheads) or (d_inner, d_state)
    _, dt = torch.split(
            xbcdt, [layer.mamba.d_inner + 2 * layer.mamba.ngroups * layer.mamba.d_state, layer.mamba.nheads], dim=-1
        )
    dt = F.softplus(dt + layer.mamba.dt_bias)  # (B, L, nheads)

    lambda_values = torch.exp(dt*A)
    lambda_values = lambda_values.detach().cpu().numpy()

    # add dimension for concatenation 
    lambda_values = np.expand_dims(lambda_values, axis=-1)

    return lambda_values

def get_eig_mamba2_LTI(x, layer):
    xbcdt = layer.mamba.in_proj(x)  # (B, L, d_in_proj)
    A = -F.softplus(layer.mamba.A)  # (nheads) or (d_inner, d_state)
    
    batch, seqlen, dim = x.shape    
    beta = layer.mamba.beta.expand(batch, seqlen, -1)

    lambda_values = torch.exp(beta*A)
    lambda_values = lambda_values.detach().cpu().numpy()

    # add dimension for concatenation 
    lambda_values = np.expand_dims(lambda_values, axis=-1)

    return lambda_values

def get_init_layers_ssm(seed,data_config, train_config, model_config, SEQ_LEN, init_fn,batch_size):
    key = random.PRNGKey(seed)
    init_rng, train_rng = random.split(key, num=2)
    
    ssm = init_fn(model_config["state_dim"], model_config["hidden_dim"], **model_config)
    
    model = partial(
            BatchClassificationModel,
            ssm=ssm,
            d_output=model_config["output_dim"],
            d_model=model_config["hidden_dim"],
            n_layers=model_config["num_layers"],
            activation=model_config["activation"],
            dropout=model_config["dropout"],
            pooling=model_config["pooling"],
            prenorm=model_config["prenorm"],
            norm=model_config["norm"],
            padded=False,
        )
    
    dummy_input = jnp.ones((batch_size, SEQ_LEN, model_config["input_dim"]))
    model = model(training=False)
    rng, dropout_rng = jax.random.split(init_rng, num=2)
    variables = model.init({"params": rng, "dropout": dropout_rng}, dummy_input)
    
    params = variables["params"]
    
    layers = []
    for layer in params["encoder"]:
        if layer.startswith("layers"):
            layers.append(params["encoder"][layer]["seq"])

    return layers

def get_trained_layers_ssm(path):
    checkpointer = ocp.PyTreeCheckpointer()
    raw_restored = checkpointer.restore(path)

    params = raw_restored["model"]["params"]
    
    layers = []
    for layer in params["encoder"]:
        if layer.startswith("layers"):
            layers.append(params["encoder"][layer]["seq"])

    return layers

def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, jnp.newaxis]
    Ct = C[jnp.newaxis, :]

    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


def get_eigvals_ssm(model, layer_list, layer_nr, idx, SEQ_LEN):
    if model in ["s4"]:
        layer = layer_list[layer_nr]
    
        step = jnp.exp(layer["log_step"][0,idx])
        Lambda_re = layer["Lambda_re"][:,idx]
        Lambda_im = layer["Lambda_im"][:,idx]
        Lambda = jnp.clip(Lambda_re, None, -1e-4) + 1j * Lambda_im
        B = layer["B"][:,idx]
        C = layer["C"][:,idx,:]
        C_tilde = C[:,0] + 1j * C[:,1]
        P = layer["P"][:,idx]
    
        Ad, Bd, Cd = discrete_DPLR(Lambda, P, P, B, C_tilde, step, SEQ_LEN)

        lambda_values = np.linalg.eigvals(Ad)

        # add dimension for concatenation 
        lambda_values = np.expand_dims(lambda_values, axis=-1)

        return lambda_values

    if model in ["s5"]:
        layer = layer_list[layer_nr]
    
        step = jnp.exp(layer["log_step"].flatten())
        Lambda_re = layer["Lambda_re"]
        Lambda_im = layer["Lambda_im"]
        Lambda = Lambda_re + 1j * Lambda_im

        lambda_values = jnp.exp(Lambda * step)

        # add dimension for concatenation 
        lambda_values = np.expand_dims(lambda_values, axis=-1)

        return lambda_values

    elif model in ["lru"]:
        layer = layer_list[layer_nr]
        
        nu_log = layer["nu_log"]
        theta_log = layer["theta_log"]

        lambda_values = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))

        # add dimension for concatenation 
        lambda_values = np.expand_dims(lambda_values, axis=-1)

        return lambda_values

    else:
        print("model type {0} is not supported!".format(model))
        return None
    
def threshold_analysis(eig_val, thresholds, num_layers, num_heads, batch_size):
    """
    eig_val: shape (B, N, num_heads, num_layers)
    thresholds: 1D array of threshold values
    """
    thresholds = thresholds.flatten()
    num_thresholds = thresholds.shape[0]
    percentages = np.empty([num_thresholds + 1, batch_size, num_heads, num_layers])

    # Values we compare against thresholds
    # Shape: (B, N, H, L)
    eta = eig_val
    count_eta_all = eta.shape[1]  # total values per head/layer

    # First bin: 0 <= x <= first threshold
    mask_low = (eta >= 0) & (eta <= thresholds[0])
    percentages[0,:,:,:] = mask_low.sum(axis=(1)) / count_eta_all * 100

    # Last bin: > last threshold
    mask_high = eta > thresholds[-1]
    percentages[-1,:,:,:] = mask_high.sum(axis=(1)) / count_eta_all * 100

    # Middle bins: thresholds[t] <= x <= thresholds[t+1]
    for t in range(num_thresholds-1):
        mask_middle = (eta >= thresholds[t]) & (eta <= thresholds[t+1])
        percentages[t+1,:,:,:] = mask_middle.sum(axis=(1)) / count_eta_all * 100

    return percentages

def threshold_analysis_ssm(eig_val, thresholds, num_layers):
    """
    eig_val: shape (B, N, num_heads, num_layers)
    thresholds: 1D array of threshold values
    """
    thresholds = thresholds.flatten()
    num_thresholds = thresholds.shape[0]
    percentages = np.empty([num_thresholds + 1, num_layers])

    # Values we compare against thresholds
    # Shape: (B, N, H, L)
    eta = eig_val
    count_eta_all = eta.shape[0]  # total values per head/layer

    # First bin: 0 <= x <= first threshold
    mask_low = (eta >= 0) & (eta <= thresholds[0])
    percentages[0,:] = mask_low.sum(axis=(0)) / count_eta_all * 100

    # Last bin: > last threshold
    mask_high = eta > thresholds[-1]
    percentages[-1,:] = mask_high.sum(axis=(0)) / count_eta_all * 100

    # Middle bins: thresholds[t] <= x <= thresholds[t+1]
    for t in range(num_thresholds-1):
        mask_middle = (eta >= thresholds[t]) & (eta <= thresholds[t+1])
        percentages[t+1,:] = mask_middle.sum(axis=(0)) / count_eta_all * 100

    return percentages

def create_file_percentage(thresholds_radius, percentage, percentage_init, percentage_mean, percentage_init_mean, percentage_std, percentage_init_std):

    batch_size = np.shape(percentage)[1]
    num_heads = np.shape(percentage)[2]
    num_layers = np.shape(percentage)[3]

    batch_selection = np.array([0,2,4,6])
    num_selected_batches = np.shape(batch_selection)[0]

    with open('percentage_file.txt', 'w') as f:

        print("threshold radius:", thresholds_radius, "\n", file=f)
        print("batch selection:", batch_selection, "\n", file=f)

        for b in range(num_selected_batches):
            for h in range(num_heads):

                for l in range(num_layers):
                    print("percentage batch dimension", batch_selection[b], "head", h,  "layer", l, "radius init: ", np.round(percentage_init[:,batch_selection[b],h,l],1), file=f)

                for l in range(num_layers):
                    print("percentage batch dimension", batch_selection[b], "head", h,  "layer", l, "radius: ", np.round(percentage[:,batch_selection[b],h,l],1), file=f)

                if b == 0:

                    for l in range(num_layers):
                        print("percentage batch mean head", h,  "layer", l, "radius init: ", np.round(percentage_init_mean[:,h,l],1), file=f)

                    for l in range(num_layers):
                        print("percentage batch mean head", h,  "layer", l, "radius: ", np.round(percentage_mean[:,h,l],1), file=f)

                    for l in range(num_layers):
                        print("percentage batch std head", h,  "layer", l, "radius init: ", np.round(percentage_init_std[:,h,l],1), file=f)

                    for l in range(num_layers):
                        print("percentage batch std head", h,  "layer", l, "radius: ", np.round(percentage_std[:,h,l],1), file=f)

                print('\n', file=f)
            print('\n', file=f)
    
    return

def create_file_percentage_ssm(thresholds_radius, thresholds_phase, percentage, percentage_init, percentage_phase, percentage_phase_init):

    num_layers = np.shape(percentage)[1]

    with open('percentage_file.txt', 'w') as f:

        print("threshold radius:", thresholds_radius, "\n", file=f)
        print("threshold phase:", thresholds_phase, "\n", file=f)

        for l in range(num_layers):
            print("percentage layer", l, "radius init: ", np.round(percentage_init[:,l],1), file=f)
        print('\n', file=f)

        for l in range(num_layers):
            print("percentage layer", l, "radius: ", np.round(percentage[:,l],1), file=f)
        print('\n', file=f)

        for l in range(num_layers):
            print("percentage layer", l, "phase init: ", np.round(percentage_phase_init[:,l],1), file=f)
        print('\n', file=f)

        for l in range(num_layers):
            print("percentage layer", l, "phase: ", np.round(percentage_phase[:,l],1), file=f)    
    
    return


def eval_eig(args, conf_args, wandb_config, data_config, loader, path_file, perf):    
    model_config = args["model"]
    train_config = args["train"]
    data_config = args["dataset"]
    seed = args["seed"]
    batch_size = conf_args["batch_size"]
    num_layers = model_config['num_layers']
    pseudoLTI = model_config["pseudoLTI"] if "pseudoLTI" in model_config else False

    if os.path.isabs(path_file):
        path = path_file
    else:  
        # get absolute path 
        path_checkpoint = os.path.abspath(os.getcwd())
        path = path_checkpoint + "/" + path_file

    # extract model class [mamba | transformer | etc.]
    layer_type = model_config.pop("layer")
    
    if layer_type in ["mamba", "transformer"]:

        num_heads = model_config['num_heads']
        torch.manual_seed(seed)

        d_model = model_config['hidden_dim']
        d_qk = model_config['state_dim']

        # extract model class
        if layer_type == "mamba":
            model_cls = Mamba
        elif layer_type == "transformer":
            model_cls = Transformer

        ## Init
    
        model = model_cls(model_config).to("cuda")
        init_layers = get_layers(layer_type, model)
    
        # load input
        encoder = model.encoder
        X, y, _ = next(iter(loader))

        # pass through encoder
        X = X.to("cuda")
        x = encoder(X)
        x_init = x

        # select model type
        if layer_type == "mamba":
            # start with layer 0
            init_layer = init_layers[0]
            x_init = init_layers[0](x_init)
            if pseudoLTI:
                eig_init = get_eig_mamba2_LTI(x_init, init_layer)
            else:
                eig_init = get_eig_mamba2(x_init, init_layer)

            # investigate all layers
            for i in range(num_layers-1):
                init_layer = init_layers[i+1]
                x_init = init_layers[i+1](x_init)
                if pseudoLTI:
                    eig_init = np.concatenate((eig_init,get_eig_mamba2_LTI(x_init, init_layer)), axis=-1)
                else:
                    eig_init = np.concatenate((eig_init,get_eig_mamba2(x_init, init_layer)), axis=-1)

        if layer_type == "transformer":

            if model_config['attention_fn'] == "sm-attention":
                # start with layer 0
                init_layer = init_layers[0]
                x_init = init_layers[0](x_init)
                eig_init = get_eig_att_softmax(x_init, init_layer, d_qk, num_heads, d_model)

                # investigate all layers
                for i in range(num_layers-1):
                    init_layer = init_layers[i+1]
                    x_init = init_layers[i+1](x_init)
                    eig_init = np.concatenate((eig_init,get_eig_att_softmax(x_init, init_layer, d_qk, num_heads, d_model)), axis=-1)

            elif model_config['attention_fn'] == "lin-attention":
                # start with layer 0
                init_layer = init_layers[0]
                x_init = init_layers[0](x_init)
                eig_init = get_eig_att_linear(x_init, init_layer, d_qk, num_heads, d_model)

                # investigate all layers
                for i in range(num_layers-1):
                    init_layer = init_layers[i+1]
                    x_init = init_layers[i+1](x_init)
                    eig_init = np.concatenate((eig_init,get_eig_att_linear(x_init, init_layer, d_qk, num_heads, d_model)), axis=-1)

            elif model_config['attention_fn'] == "norm-attention":
                # start with layer 0
                init_layer = init_layers[0]
                x_init = init_layers[0](x_init)
                eig_init = get_eig_att_norm(x_init, init_layer, d_qk, num_heads, d_model, model_config)

                # investigate all layers
                for i in range(num_layers-1):
                    init_layer = init_layers[i+1]
                    x_init = init_layers[i+1](x_init)
                    eig_init = np.concatenate((eig_init,get_eig_att_norm(x_init, init_layer, d_qk, num_heads, d_model, model_config)), axis=-1)


        ## Trained
        # load weights of trained model
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()

        trained_layers = get_layers(layer_type, model)

        # load input
        encoder = model.encoder
        X, y, _ = next(iter(loader))

        # pass through encoder
        X = X.to("cuda")
        x = encoder(X)
        x_init = x

        # select model type
        if layer_type == "mamba":
            # start with layer 0
            trained_layer = trained_layers[0]
            x = trained_layers[0](x)
            if pseudoLTI:
                eig = get_eig_mamba2_LTI(x, trained_layer)
            else:
                eig = get_eig_mamba2(x, trained_layer)

            # investigate all layers
            for i in range(num_layers-1):
                trained_layer = trained_layers[i+1]
                x = trained_layers[i+1](x)
                if pseudoLTI:
                    eig = np.concatenate((eig,get_eig_mamba2_LTI(x, trained_layer)), axis=-1)
                else:
                    eig = np.concatenate((eig,get_eig_mamba2(x, trained_layer)), axis=-1)


            thresholds_radius = np.array([0.1,0.5,0.9,1.0,10,100])

            eig_rad_init = np.sqrt(np.power(eig_init.real,2) + np.power(eig_init.imag,2))
            eig_rad = np.sqrt(np.power(eig.real,2) + np.power(eig.imag,2))

            
            percentage_init = threshold_analysis(eig_rad_init, thresholds_radius, num_layers, num_heads, batch_size)
            percentage = threshold_analysis(eig_rad, thresholds_radius, num_layers, num_heads, batch_size)

            thresholds_phase = np.array([1,10,45,90,180])

            eig_phase_init = np.arctan2(eig_init.imag, eig_init.real) * 180 / np.pi
            eig_phase = np.arctan2(eig.imag, eig.real) * 180 / np.pi

            percentage_phase_init = threshold_analysis(eig_phase_init, thresholds_phase, num_layers, num_heads, batch_size)
            percentage_phase = threshold_analysis(eig_phase, thresholds_phase, num_layers, num_heads, batch_size)

            percentage_init_mean = np.mean(percentage_init, axis=1)
            percentage_init_std = np.std(percentage_init, axis=1)
            percentage_mean = np.mean(percentage, axis=1)
            percentage_std = np.std(percentage, axis=1)

            create_file_percentage(thresholds_radius, percentage, percentage_init, percentage_mean, percentage_init_mean, percentage_std, percentage_init_std)

        if layer_type == "transformer":

            if model_config['attention_fn'] == "sm-attention":
                # start with layer 0
                trained_layer = trained_layers[0]
                x = trained_layers[0](x)
                eig = get_eig_att_softmax(x, trained_layer, d_qk, num_heads, d_model)

                # investigate all layers
                for i in range(num_layers-1):
                    trained_layer = trained_layers[i+1]
                    x = trained_layers[i+1](x)
                    eig = np.concatenate((eig,get_eig_att_softmax(x, trained_layer, d_qk, num_heads, d_model)), axis=-1)

            elif model_config['attention_fn'] == "lin-attention":
                # start with layer 0
                trained_layer = trained_layers[0]
                x = trained_layers[0](x)
                eig = get_eig_att_linear(x, trained_layer, d_qk, num_heads, d_model)

                # investigate all layers
                for i in range(num_layers-1):
                    trained_layer = trained_layers[i+1]
                    x = trained_layers[i+1](x)
                    eig = np.concatenate((eig,get_eig_att_linear(x, trained_layer, d_qk, num_heads, d_model)), axis=-1)

            elif model_config['attention_fn'] == "norm-attention":
                # start with layer 0
                trained_layer = trained_layers[0]
                x = trained_layers[0](x)
                eig = get_eig_att_norm(x, trained_layer, d_qk, num_heads, d_model, model_config)

                # investigate all layers
                for i in range(num_layers-1):
                    trained_layer = trained_layers[i+1]
                    x = trained_layers[i+1](x)
                    eig = np.concatenate((eig,get_eig_att_norm(x, trained_layer, d_qk, num_heads, d_model, model_config)), axis=-1)

            thresholds_radius = np.array([0.1,0.5,0.9,1.0,10,100])

            
            percentage_init = threshold_analysis(eig_init, thresholds_radius, num_layers, num_heads, batch_size)
            percentage = threshold_analysis(eig, thresholds_radius, num_layers, num_heads, batch_size)

            thresholds_phase = np.array([1,10,45,90,180])

            percentage_phase_init = threshold_analysis(0*eig_init, thresholds_phase, num_layers, num_heads, batch_size)
            percentage_phase = threshold_analysis(0*eig, thresholds_phase, num_layers, num_heads, batch_size)
            

            percentage_init_mean = np.mean(percentage_init, axis=1)
            percentage_init_std = np.std(percentage_init, axis=1)
            percentage_mean = np.mean(percentage, axis=1)
            percentage_std = np.std(percentage, axis=1)

            create_file_percentage(thresholds_radius, percentage, percentage_init, percentage_mean, percentage_init_mean, percentage_std, percentage_init_std)

    elif layer_type in ["lru", "s4", "s5"]:

        SEQ_LEN = model_config["seq_len"]
        num_heads = 1

        dim_idx = 1 # check what makes sense here
        if layer_type == "lru":
            LRU_init_layers = get_init_layers_ssm(seed,data_config, train_config, model_config, SEQ_LEN, init_LRU,batch_size)
            eig_init = get_eigvals_ssm("lru", LRU_init_layers, 0, dim_idx, SEQ_LEN)
            for i in range(num_layers-1):
                eig_init = np.concatenate((eig_init,get_eigvals_ssm("lru", LRU_init_layers, i+1, dim_idx, SEQ_LEN)), axis=-1)

            LRU_trained_layers = get_trained_layers_ssm(path)
            eig = get_eigvals_ssm("lru", LRU_trained_layers, 0, dim_idx, SEQ_LEN)
            for i in range(num_layers-1):
                eig = np.concatenate((eig,get_eigvals_ssm("lru", LRU_trained_layers, i+1, dim_idx, SEQ_LEN)), axis=-1)

        
        elif layer_type == "s5":
            S5_init_layers = get_init_layers_ssm(seed,data_config, train_config, model_config, SEQ_LEN, init_S5,batch_size)
            eig_init = get_eigvals_ssm("s5", S5_init_layers, 0, dim_idx, SEQ_LEN)
            for i in range(num_layers-1):
                eig_init = np.concatenate((eig_init,get_eigvals_ssm("s5", S5_init_layers, i+1, dim_idx, SEQ_LEN)), axis=-1)

            S5_trained_layers = get_trained_layers_ssm(path)
            eig = get_eigvals_ssm("s5", S5_trained_layers, 0, dim_idx, SEQ_LEN)
            for i in range(num_layers-1):
                eig = np.concatenate((eig,get_eigvals_ssm("s5", S5_trained_layers, i+1, dim_idx, SEQ_LEN)), axis=-1)
        
        elif layer_type == "s4":
            S4_init_layers = get_init_layers_ssm(seed,data_config, train_config, model_config, SEQ_LEN, init_S4,batch_size)
            eig_init = get_eigvals_ssm("s4", S4_init_layers, 0, dim_idx, SEQ_LEN)
            for i in range(num_layers-1):
                eig_init = np.concatenate((eig_init,get_eigvals_ssm("s4", S4_init_layers, i+1, dim_idx, SEQ_LEN)), axis=-1)
            
            S4_trained_layers = get_trained_layers_ssm(path)
            eig = get_eigvals_ssm("s4", S4_trained_layers, 0, dim_idx, SEQ_LEN)
            for i in range(num_layers-1):
                eig = np.concatenate((eig,get_eigvals_ssm("s4", S4_trained_layers, i+1, dim_idx, SEQ_LEN)), axis=-1)

        thresholds_radius = np.array([0.1,0.5,0.9,1.0,10,100])

        eig_rad_init = np.sqrt(np.power(eig_init.real,2) + np.power(eig_init.imag,2))
        eig_rad = np.sqrt(np.power(eig.real,2) + np.power(eig.imag,2))

        percentage_init = threshold_analysis_ssm(eig_rad_init, thresholds_radius, num_layers)
        percentage = threshold_analysis_ssm(eig_rad, thresholds_radius, num_layers)

        thresholds_phase = np.array([1,10,45,90,180])

        eig_phase_init = np.arctan2(eig_init.imag, eig_init.real) * 180 / np.pi
        eig_phase = np.arctan2(eig.imag, eig.real) * 180 / np.pi

        percentage_phase_init = threshold_analysis_ssm(eig_phase_init, thresholds_phase, num_layers)
        percentage_phase = threshold_analysis_ssm(eig_phase, thresholds_phase, num_layers)

        percentage_init_mean = 0
        percentage_init_std = 0
        percentage_mean = 0
        percentage_std = 0

        create_file_percentage_ssm(thresholds_radius, thresholds_phase, percentage, percentage_init, percentage_phase, percentage_phase_init)

    else:
        raise RuntimeError("{0} is not a valid model option".format(layer_type))
    
    path_to_percentage_file = os.path.abspath(os.getcwd()) + "/percentage_file.txt"

    if wandb_config is not None:
        print("Saving artifact on W&B....")
        dim_conv = model_config["dim_conv"] if "dim_conv" in model_config else 0
        name_model_no_perf = data_config["name"] + "{0}-dmodel{1}-seed{4}-num_layers{5}-dqk{2}-conv_dim{6}-lr{3}".format(wandb_config["name"], model_config["hidden_dim"], model_config["state_dim"], train_config["lr"], args["seed"], model_config["num_layers"], dim_conv)
        name_model = name_model_no_perf + "-perf{0:0.3f}".format(perf)
        
        run = wandb.init(group="artifact_upload", entity=wandb_config["entity"], project=wandb_config["project"], name="upload"+name_model, job_type="add-dataset")
        artifact = wandb.Artifact(name="eigen_values_" + name_model_no_perf, type="dataset")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "eig.npy")
            path2 = os.path.join(tmpdir, "eig_init.npy")

            
            path3 = os.path.join(tmpdir, "percentage.npy")
            path4 = os.path.join(tmpdir, "percentage_init.npy")
            path5 = os.path.join(tmpdir, "percentage_phase.npy")
            path6 = os.path.join(tmpdir, "percentage_phase_init.npy")
            path7 = os.path.join(tmpdir, "percentage_mean.npy")
            path8 = os.path.join(tmpdir, "percentage_init_mean.npy")
            path9 = os.path.join(tmpdir, "percentage_std.npy")
            path10 = os.path.join(tmpdir, "percentage_init_std.npy")

            path_to_full_config = os.path.join(tmpdir, "used_config.yaml")

            np.save(path1, eig)
            np.save(path2, eig_init)            
            
            np.save(path3, percentage)
            np.save(path4, percentage_init)
            np.save(path5, percentage_phase)
            np.save(path6, percentage_phase_init)
            np.save(path7, percentage_mean)
            np.save(path8, percentage_init_mean)
            np.save(path9, percentage_std)
            np.save(path10, percentage_init_std)

            with open(path_to_full_config, "w") as file:
                yaml.dump(args, file, default_flow_style=False, sort_keys=False)
        
            artifact.add_file(local_path=path1, name="eigen_values_" + name_model)
            artifact.add_file(local_path=path2, name="eigen_values_init_" + name_model)
            artifact.add_file(local_path=path3, name="percentage_" + name_model)
            artifact.add_file(local_path=path4, name="percentage_init_" + name_model)
            artifact.add_file(local_path=path5, name="percentage_phase_" + name_model)
            artifact.add_file(local_path=path6, name="percentage_pase_init_" + name_model)
            artifact.add_file(local_path=path7, name="percentage_mean_" + name_model)
            artifact.add_file(local_path=path8, name="percentage_init_mean_" + name_model)
            artifact.add_file(local_path=path9, name="percentage_std_" + name_model)
            artifact.add_file(local_path=path10, name="percentage_init_std_" + name_model)
            artifact.add_file(local_path=path_to_percentage_file, name="percentage_file_" + name_model)
            artifact.add_file(local_path=path_to_full_config, name="used_config-" + name_model)


            artifact.save()
    else:
        print("Saving artifact locally....")
        save_path = conf_args["save_path"] if "save_path" in conf_args else ""
        dim_conv = model_config["dim_conv"] if "dim_conv" in model_config else 0
        name_model_no_perf = data_config["name"] + "dmodel{0}-seed{3}-num_layers{4}-dqk{1}-conv_dim{5}-lr{2}".format(model_config["hidden_dim"], model_config["state_dim"], train_config["lr"], args["seed"], model_config["num_layers"], dim_conv)
        name_model = name_model_no_perf + "-perf{0:0.3f}".format(perf)
                
        tmpdir = save_path + name_model
        path1 = os.path.join(tmpdir, "eig.npy")
        path2 = os.path.join(tmpdir, "eig_init.npy")
        path3 = os.path.join(tmpdir, "percentage.npy")
        path4 = os.path.join(tmpdir, "percentage_init.npy")
        path5 = os.path.join(tmpdir, "percentage_phase.npy")
        path6 = os.path.join(tmpdir, "percentage_phase_init.npy")
        path7 = os.path.join(tmpdir, "percentage_mean.npy")
        path8 = os.path.join(tmpdir, "percentage_init_mean.npy")
        path9 = os.path.join(tmpdir, "percentage_std.npy")
        path10 = os.path.join(tmpdir, "percentage_init_std.npy")

        path_to_full_config = os.path.join(tmpdir, "used_config.yaml")
        directory_name = tmpdir

        try:
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        np.save(path1, eig)
        np.save(path2, eig_init)            
        np.save(path3, percentage)
        np.save(path4, percentage_init)
        np.save(path5, percentage_phase)
        np.save(path6, percentage_phase_init)
        np.save(path7, percentage_mean)
        np.save(path8, percentage_init_mean)
        np.save(path9, percentage_std)
        np.save(path10, percentage_init_std)

        with open(path_to_full_config, "w") as file:
            yaml.dump(args, file, default_flow_style=False, sort_keys=False)
    try:
        if wandb_config is not None:
            wandb.finish()
        return eig, eig_init, percentage, percentage_init, percentage_phase, percentage_phase_init
    except:    
        return eig, eig_init, percentage, percentage_init, percentage_phase, percentage_phase_init

    
