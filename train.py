import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pytorch_warmup as warmup
import wandb
from tqdm import tqdm
from functools import partial
import jax.numpy as jnp
from jax import random
import sys
import time
import jax

from jax_helpers import (
    create_train_state,
    create_train_state_s5,
    reduce_lr_on_plateau,
    linear_warmup,
    cosine_annealing,
    constant_lr,
    train_epoch,
    prep_batch,
    train_step,
    cross_entropy_loss,
    update_learning_rate_per_step,
    save_model,
    validate,
    eval_step
)

from models import Mamba, Transformer, BatchClassificationModel, init_LRU, init_S5, init_S4

def train_torch_step(seed, trainloader, testloader, model_cls, metrics_fn, wandb_config, train_config, model_config, checkpoint):
    torch.manual_seed(seed)
    device = "cuda"
    model = model_cls(model_config).to(device)
    if "use_flash" in model_config:
        use_mixed_precision = model_config["use_flash"] and model_config["attention_fn"] in ["sm-attention", "jamba"]
    else:
        use_mixed_precision = False

    # log model parameters
    nr_params = sum(p.numel() for p in model.parameters())
    print("Nr. of parameters: {0}".format(nr_params))

    nr_params_encoder = sum(p.numel() for p in model.encoder.parameters())
    print("Nr. of parameters in the encoder: {0}".format(nr_params_encoder))
    if wandb_config is not None:
        wandb.log({"params": nr_params})
        wandb.log({"params without encoder": nr_params - nr_params_encoder})
    
    # Initialize optimizer
    if train_config["param_group"] is not None:
        group_params = [p for name, p in model.named_parameters() if train_config["param_group"] in name]
        params = [p for name, p in model.named_parameters() if train_config["param_group"] not in name]
        optimizer = torch.optim.AdamW(params, lr=train_config["lr"], weight_decay=train_config["wd"], betas=train_config["betas"] if "betas" in train_config else (0.9, 0.999))
        group_optimizer = torch.optim.AdamW(group_params, lr=train_config["group_lr"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["wd"], betas=train_config["betas"] if "betas" in train_config else (0.9, 0.999))
        group_optimizer = None
            
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["total_steps"], eta_min = 5e-6)
    if group_optimizer is not None:
        group_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(group_optimizer, T_max=train_config["total_steps"], eta_min = 5e-6)
    if "warmup_steps" in train_config:
        warmup_scheduler = warmup.LinearWarmup(optimizer, train_config["warmup_steps"])
    else:
        warmup_scheduler = None
    
    if use_mixed_precision: 
        scaler = GradScaler() 

    running_loss = 0.0
    running_performance = 0.0
    step = 0
    stop_training = False
    total_steps = train_config["total_steps"]
    eval_every = train_config["eval_every"]

    # Create progress bar for total steps
    pbar = tqdm(total=total_steps, desc="Training steps")

    while step < total_steps and not stop_training:
        if group_optimizer is not None:
            group_optimizer.zero_grad()

        for X, y, _ in trainloader:
            if step >= total_steps:
                break
            
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)

            # Calculate loss
            if use_mixed_precision:
                with autocast():
                    y_hat = model(X)
                    loss = F.cross_entropy(y_hat.reshape(-1, y_hat.size(-1)), y.reshape(-1))
            else:
                y_hat = model(X)
                loss = F.cross_entropy(y_hat.reshape(-1, y_hat.size(-1)), y.reshape(-1))
            running_loss += loss.item()
            running_performance += metrics_fn(y_hat, y)
            
            # Backward pass
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # update convex combination learning rate
            if group_optimizer is not None:
                if step % train_config["update_step"] == 0 and step > 0:
                    group_optimizer.step()
                    group_optimizer.zero_grad()

             # Update learning rate schedulers
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    scheduler.step()
            if group_optimizer is not None:
                group_scheduler.step()
            
            # Evaluate model and log results
            if step % eval_every == 0 and step > 0:
                train_loss = running_loss/eval_every
                tqdm.write("\nLoss: {0:.3f}".format(train_loss))
                train_perf = running_performance/eval_every
                tqdm.write("Train performance: {0:.4f}".format(train_perf))

                # reset running loss and performance
                running_loss = 0.0
                running_performance = 0.0

                # Evaluate model
                model.eval()
                test_performance = 0.0
                test_loss = 0.0
                with torch.no_grad():
                    for X, y, _ in testloader:
                        X = X.to(device)
                        y = y.to(device)
                        y_hat = model(X)

                        loss = F.cross_entropy(y_hat.reshape(-1, y_hat.size(-1)), y.reshape(-1))
                        test_loss += loss.item()
                        test_performance += metrics_fn(y_hat, y)
                    test_loss = test_loss/len(testloader)
                    test_perf = test_performance / len(testloader)
                    tqdm.write("Test performance: {0:.4f}".format(test_perf))
                
                if wandb_config is not None:
                    # Log results
                    if group_optimizer is not None:
                        wandb.log(
                            {"train perf": train_perf,
                            "test perf": test_perf,
                            "train loss": train_loss,
                            "test loss": test_loss,
                            "lr": optimizer.param_groups[0]['lr'],
                            "alpha lr": group_optimizer.param_groups[0]['lr']}
                            )
                        
                        if model_config["mixer"] in ["hybrid"]:
                            log_dict = {}
                            for i in range(len(model.layers)):
                                a = model.layers[i].mixer.alpha
                                a = a.detach().cpu()
                                log_dict["mixer_alpha_{0}".format(i)] = F.sigmoid(a).numpy().item()
                            wandb.log(log_dict)
                        
                        if "mode" in model_config:
                            if model_config["mode"] in ["hybrid"]:
                                log_dict = {}
                                for i in range(len(model.layers)):
                                    a = model.layers[i].attention.inner_attn.alpha
                                    a = a.detach().cpu()
                                    log_dict["alpha_{0}".format(i)] = F.sigmoid(a).numpy().item()
                                wandb.log(log_dict)
                    else:
                        wandb.log(
                            {"train perf": train_perf,
                            "test perf": test_perf,
                            "train loss": train_loss,
                            "test loss": test_loss,
                            "lr": optimizer.param_groups[0]['lr']}
                            )
                    
                model.train()

                # check early stopping criterion
                if "stop_criterion" in train_config:
                    if test_perf > train_config["stop_criterion"]:
                        tqdm.write("Stopping training as test performance is greater than stopping criterion {0:.2f}".format(train_config["stop_criterion"]))
                        sys.stdout.flush()
                        stop_training = True
                        break
            
            step += 1
            sys.stdout.flush()
            pbar.update(1)
    pbar.close()

    path = None

    if checkpoint is not None:
        torch.save(model.state_dict(), checkpoint + "-perf{0:0.3f}.pth".format(test_perf))
        path = checkpoint + "-perf{0:0.3f}.pth".format(test_perf)
    
    return path, test_perf

def train_torch(seed, trainloader, testloader, model_cls, metrics_fn, wandb_config, train_config, model_config, checkpoint):
    torch.manual_seed(seed)
    device = "cuda"
    model = model_cls(model_config).to(device)
    if "use_flash" in model_config:
        use_mixed_precision = model_config["use_flash"] and model_config["attention_fn"] in ["sm-attention", "jamba"]
    else:
        use_mixed_precision = False

    # log model parameters
    nr_params = sum(p.numel() for p in model.parameters())
    print("Nr. of parameters: {0}".format(nr_params))

    nr_params_encoder = sum(p.numel() for p in model.encoder.parameters())
    print("Nr. of parameters in the encoder: {0}".format(nr_params_encoder))
    if wandb_config is not None:
        wandb.log({"params": nr_params})
        wandb.log({"params without encoder": nr_params - nr_params_encoder})
    
    # Initialize optimizer
    if train_config["param_group"] is not None:
        group_params = [p for name, p in model.named_parameters() if train_config["param_group"] in name]
        params = [p for name, p in model.named_parameters() if train_config["param_group"] not in name]
        optimizer = torch.optim.AdamW(params, lr=train_config["lr"], weight_decay=train_config["wd"], betas=train_config["betas"] if "betas" in train_config else (0.9, 0.999))
        group_optimizer = torch.optim.AdamW(group_params, lr=train_config["group_lr"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["wd"], betas=train_config["betas"] if "betas" in train_config else (0.9, 0.999))
        group_optimizer = None
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"], eta_min = 5e-6)
    if group_optimizer is not None:
        group_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(group_optimizer, T_max=train_config["num_epochs"], eta_min = 5e-6)
    if "warmup" in train_config:
        warmup_scheduler = warmup.LinearWarmup(optimizer, train_config["warmup"])
    else:
        warmup_scheduler = None

    if use_mixed_precision:
        scaler = GradScaler()

    # Training loop
    for epoch in range(train_config["num_epochs"]):
        if group_optimizer is not None:
            group_optimizer.zero_grad()
            step = 0

        running_loss = 0.0
        running_performance = 0.0
        if epoch == 0:
            start_time = time.time()
        for X, y, _ in tqdm(trainloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            
            # Calculate loss
            if use_mixed_precision:
                with autocast():
                    y_hat = model(X)
                    loss = torch.nn.functional.cross_entropy(y_hat, y)
            else:
                y_hat = model(X)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item()
            running_performance += metrics_fn(y_hat, y)

            # Backward pass
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # update convex combination learning rate
            if group_optimizer is not None:
                step += 1
                if step % train_config["update_step"] == 0 and step > 0:
                    group_optimizer.step()
                    group_optimizer.zero_grad()
        
        train_loss = running_loss/len(trainloader)
        tqdm.write("Loss: {0:.3f}".format(train_loss))
        train_perf = running_performance / len(trainloader)
        tqdm.write("Train performance: {0:.4f}".format(train_perf))

        # Update learning rate scheduler
        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                scheduler.step()
        if group_optimizer is not None:
            group_scheduler.step()

        # evaluate model
        model.eval()
        test_performance = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(testloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                test_loss += loss.item()
                test_performance += metrics_fn(y_hat, y)

        test_loss = test_loss/len(testloader)
        test_perf = test_performance / len(testloader)
        tqdm.write("Test performance: {0:.4f}\n".format(test_perf))

        if epoch == 0:
            tqdm.write("Estimated time to completion: {0:.2f} hours\n".format((train_config["num_epochs"] - 1) * (time.time() - start_time) / 3600))

        if wandb_config is not None:
            if group_optimizer is not None:
                wandb.log(
                    {"train perf": train_perf,
                    "test perf": test_perf,
                    "train loss": train_loss,
                    "test loss": test_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "alpha lr": group_optimizer.param_groups[0]['lr']}
                    )
                
                if model_config["mixer"] in ["hybrid"]:
                    log_dict = {}
                    for i in range(len(model.layers)):
                        a = model.layers[i].mixer.alpha
                        a = a.detach().cpu()
                        log_dict["mixer_alpha_{0}".format(i)] = F.sigmoid(a).numpy().item()
                    wandb.log(log_dict)
                
                if "mode" in model_config:
                    if model_config["mode"] in ["hybrid"]:
                        log_dict = {}
                        for i in range(len(model.layers)):
                            a = model.layers[i].attention.inner_attn.alpha
                            a = a.detach().cpu()
                            log_dict["alpha_{0}".format(i)] = F.sigmoid(a).numpy().item()
                        wandb.log(log_dict)
            else:
                wandb.log(
                    {"train perf": train_perf,
                    "test perf": test_perf,
                    "train loss": train_loss,
                    "test loss": test_loss,
                    "lr": optimizer.param_groups[0]['lr']}
                    )
        model.train()

        # check early stopping criterion
        if "stop_criterion" in train_config:
            if test_perf > train_config["stop_criterion"]:
                tqdm.write("Stopping training as test performance is greater than stopping criterion {0:.2f}".format(train_config["stop_criterion"]))
                sys.stdout.flush()
                break

    path = None

    if checkpoint is not None:
        torch.save(model.state_dict(), checkpoint + "-perf{0:0.3f}.pth".format(test_perf))
        path = checkpoint + "-perf{0:0.3f}.pth".format(test_perf)

    return path, test_perf

def train_jax(seed, trainloader, testloader, model_init, metrics_fn, wandb_config, train_config, model_config, checkpoint, data_config):

    key = random.PRNGKey(seed)
    init_rng, train_rng = random.split(key, num=2)

    ssm = model_init(model_config["state_dim"], model_config["hidden_dim"], **model_config)

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
        padded=train_config["padded"],
    )

    if not "betas" in train_config or train_config["betas"] is None:
        train_config["betas"] = [0.9, 0.999]

    # Initialize training state

    if model_config["layer"] == "s5":
        create_train_state_general = create_train_state_s5
    else:
        create_train_state_general = create_train_state

    state, (nr_params, encoder_params) = create_train_state_general(
        model,
        init_rng,
        in_dim=model_config["input_dim"],
        batch_size=train_config["batch_size"],
        seq_len=model_config["seq_len"],
        weight_decay=train_config["wd"],
        norm=model_config["norm"],
        ssm_lr=train_config["ssm_lr"],
        ssm_vars=model_config["ssm_lr_vars"],
        lr=train_config["lr"],
        padded=train_config["padded"],
        betas=train_config["betas"]
    )

    print("Nr. of parameters: {0}".format(nr_params))
    print("Nr. of parameters in the encoder: {0}".format(encoder_params))
    if wandb_config is not None:
        wandb.log({"params": nr_params})
        wandb.log({"params without encoder": nr_params - encoder_params})

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_config["train_size"] / train_config["batch_size"])
    lr, ssm_lr = train_config["lr"], train_config["ssm_lr"] # init learning rates
    warmup = train_config["warmup"] if "warmup" in train_config else 0
    for epoch in range(train_config["num_epochs"]):
        print(f"[*] Starting Training Epoch {epoch + 1} out of " +  str(train_config["num_epochs"]) + "...")
        if epoch < warmup:
            print("Using linear warmup for epoch {}".format(epoch + 1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * warmup
        elif train_config["cosine_anneal"]:
            print("Using cosine annealing for epoch {}".format(epoch + 1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * train_config["num_epochs"] - (steps_per_epoch * warmup)
        else:
            print("Using constant lr for epoch {}".format(epoch + 1))
            decay_function = constant_lr
            end_step = None

        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, train_config["lr_min"])

        train_rng, skey = random.split(train_rng)
        state, train_loss, step = train_epoch(
            state, skey, model, trainloader, model_config["seq_len"], model_config["input_dim"], model_config["norm"], lr_params
        )

        print(f"[*] Running Epoch {epoch + 1} Test...")
        val_loss, val_acc = validate(state, model, testloader, model_config["seq_len"], model_config["input_dim"], model_config["norm"])

        print(f"\n=>> Epoch {epoch + 1} Metrics ===")
        print(
            f"\tTrain Loss: {train_loss:.5f}  -- Test Loss: {val_loss:.5f}\n"
            f"\tTest Accuracy: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(
            input, factor=train_config["reduce_factor"], patience=train_config["lr_patience"], lr_min=train_config["lr_min"]
        )

        # Print best accuracy & loss so far...
        print(
            f"\tBest Test Loss: {best_loss:.5f} -- Best Test Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if wandb_config is not None:
            metrics = {
                "train loss": train_loss,
                "test loss": val_loss,
                "test perf": val_acc,
                "lr": state.opt_state.inner_states["regular"].inner_state.hyperparams["learning_rate"],
                "ssm_lr": state.opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"],
            }
            wandb.log(metrics)
            wandb.run.summary["Best Val Loss"] = best_loss
            wandb.run.summary["Best Val Accuracy"] = best_acc
            wandb.run.summary["Best Epoch"] = best_epoch

    path = None

    if checkpoint is not None:
        config = {"model": model_config, "train": train_config, "data": data_config}
        ckpt = {"model": state, "config": config}
        save_model(checkpoint + "-perf{0:0.3f}".format(val_acc), ckpt)
        path = checkpoint + "-perf{0:0.3f}".format(val_acc)

    return path, val_acc

def train_jax_step(seed, trainloader, testloader, model_init, metrics_fn, wandb_config, train_config, model_config, checkpoint, data_config):

    key = random.PRNGKey(seed)
    init_rng, train_rng = random.split(key, num=2)

    if not "betas" in train_config or train_config["betas"] is None:
        train_config["betas"] = [0.9, 0.999]

    ssm = model_init(model_config["state_dim"], model_config["hidden_dim"], **model_config)

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
        padded=train_config["padded"],
    )

    if model_config["layer"] == "s5":
        create_train_state_general = create_train_state_s5
    else:
        create_train_state_general = create_train_state

    # Initialize training state
    state, (nr_params, encoder_params) = create_train_state_general(
        model,
        init_rng,
        in_dim=model_config["input_dim"],
        batch_size=train_config["batch_size"],
        seq_len=model_config["seq_len"],
        weight_decay=train_config["wd"],
        norm=model_config["norm"],
        ssm_lr=train_config["ssm_lr"],
        ssm_vars=model_config["ssm_lr_vars"],
        lr=train_config["lr"],
        padded=train_config["padded"],
        betas=train_config["betas"]
    )
    
    print("Nr. of parameters: {0}".format(nr_params))
    print("Nr. of parameters in the encoder: {0}".format(encoder_params))
    if wandb_config is not None:
        wandb.log({"params": nr_params})
        wandb.log({"params without encoder": nr_params - encoder_params})

    # Training Loop over epochs
    best_loss, best_acc, best_eval_round = 100000000, -100000000.0, 0  # This best loss is val_loss
    
    running_loss = 0.0
    running_performance = 0.0
    step = 0
    step_lr = 0
    stop_training = False
    total_steps = train_config["total_steps"]
    eval_every = train_config["eval_every"]

    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    # steps_per_epoch = int(train_config["train_size"] / train_config["batch_size"])
    lr, ssm_lr = train_config["lr"], train_config["ssm_lr"] # init learning rates
    warmup = train_config["warmup_steps"] if "warmup_steps" in train_config else 0

    # Create progress bar for total steps
    pbar = tqdm(total=total_steps, desc="Training steps")
    print_warmup = True
    print_lr_scheduler = True
    model1 = model(training=True)  # model in training mode
    model2 = model(training=False) 

    while step < total_steps and not stop_training:

        train_rng, rng = random.split(train_rng)
        for batch in trainloader:
            if step >= total_steps:
                break
            if step < warmup:
                if print_warmup:
                    print("Using linear warmup for step {}".format(step + 1))
                    print_warmup = False
                decay_function = linear_warmup
                end_step = warmup
            elif train_config["cosine_anneal"]:
                if print_lr_scheduler:
                    print("Using cosine annealing for step {}".format(step + 1))
                    print_lr_scheduler = False
                decay_function = cosine_annealing
                # for per step learning rate decay
                end_step = total_steps - warmup
            else:
                print("Using constant lr for step {}".format(step + 1))
                decay_function = constant_lr
                end_step = None

            #  Passing this around to manually handle per step learning rate decay.
            lr_params = (decay_function, ssm_lr, lr, step_lr, end_step, train_config["lr_min"])
            seq_len = model_config["seq_len"]
            in_dim =  model_config["input_dim"]
            norm = model_config["norm"]
            
            inputs, labels = prep_batch(batch, seq_len, in_dim, data_config["_name_"] != "listops" )

            rng, drop_rng = jax.random.split(rng)  # for dropout
            state, loss = train_step(state, drop_rng, inputs, labels, model1, norm)
            running_loss += loss

            state, step_lr = update_learning_rate_per_step(lr_params, state)

            # Evaluate model and log results
            if step % eval_every == 0 and step > 0:
                train_loss = running_loss/eval_every
                tqdm.write("\nLoss: {0:.3f}".format(train_loss))
                # reset running loss and performance
                running_loss = 0.0
                running_performance = 0.0

                # Evaluate model
                losses, accuracies = jnp.array([]), jnp.array([])

                for batch in tqdm(testloader):
                    inputs, labels = prep_batch(batch, seq_len, in_dim, data_config["_name_"] != "listops")
                    if norm == "batch":
                        logits = model2.apply({"params": state.params, "batch_stats": state.batch_stats}, inputs)
                    else:
                        logits = model2.apply({"params": state.params}, inputs)
                    loss_ev = jnp.mean(cross_entropy_loss(logits, labels))                   
                    if data_config["_name_"] != "listops":
                        acc_ev = metrics_fn(logits, labels)
                    else:
                        loss, acc_ev, logits = eval_step(inputs, labels, state, model2, norm)
                    losses = jnp.append(losses, jnp.mean(loss_ev))
                    accuracies = jnp.append(accuracies, acc_ev)
                test_loss = jnp.mean(losses)
                test_perf = jnp.mean(accuracies)
                
                tqdm.write("\nTest performance: {0:.4f}".format(test_perf))
            
                # Log results
                if wandb_config is not None:
                    metrics = {
                        "train loss": train_loss,
                        "test loss": test_loss,
                        "test perf": test_perf,
                        "lr": state.opt_state.inner_states["regular"].inner_state.hyperparams["learning_rate"],
                        "ssm_lr": state.opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"],
                    }
                    wandb.log(metrics)


                # For learning rate decay purposes:
                input = lr, ssm_lr, lr_count, test_perf, opt_acc
                lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(
                    input, factor=train_config["reduce_factor"], patience=train_config["lr_patience"], lr_min=train_config["lr_min"]
                )

                # check early stopping criterion
                if "stop_criterion" in train_config:
                    if test_perf > train_config["stop_criterion"]:
                        tqdm.write("\nStopping training as test performance is greater than stopping criterion {0:.2f}".format(train_config["stop_criterion"]))
                        sys.stdout.flush()
                        stop_training = True
                        break    

            step += 1
            sys.stdout.flush()
            pbar.update(1)

    pbar.close()

    path = None

    if checkpoint is not None:
        config = {"model": model_config, "train": train_config, "data": data_config}
        ckpt = {"model": state, "config": config}
        save_model(checkpoint + "-perf{0:0.3f}".format(test_perf), ckpt)
        path = checkpoint + "-perf{0:0.3f}".format(test_perf)

    return path, test_perf

def train(args, wandb_config, trainloader, testloader, metrics_fn):
    
    train_config = args["train"]
    model_config = args["model"]
    data_config = args["dataset"]
    dim_conv = model_config["dim_conv"] if "dim_conv" in model_config else 0
    if "save" in args and args["save"] is not None:
        checkpoint = args["save"] + "-seed-" + str(args["seed"]) + "-layers-" + str(model_config["num_layers"]) + "dim_conv" + str(dim_conv) + "-s_d-" + str(model_config["state_dim"])
    else:
        checkpoint = None
    # start wandb logging
    if wandb_config is not None:
        wandb.login(key=wandb_config["key"])
        wandb.init(
                group=wandb_config["group"],
                name="{0}-dmodel{1}-seed{4}-num_layers{5}-dqk{2}-lr{3}".format(wandb_config["name"], model_config["hidden_dim"], model_config["state_dim"], train_config["lr"], args["seed"], model_config["num_layers"]),
                entity=wandb_config["entity"],
                project=wandb_config["project"],
                config=args,
                job_type="train",
        )

    # extract model class [mamba | transformer | etc.]
    layer = model_config["layer"]
    
    # extract model class
    if layer == "mamba":
        model_cls = Mamba
    elif layer == "transformer":
        model_cls = Transformer
    elif layer == "lru":
        model_cls = init_LRU
    elif layer == "s5":
        model_cls = init_S5
    elif layer == "s4":
        model_cls = init_S4
    else:
        raise RuntimeError("{0} is not a valid model option".format(layer))
    
    if layer in ["mamba", "transformer"]:
        if args["lang_model"]:
            path, perf = train_torch_step(
                args["seed"],
                trainloader,
                testloader,
                model_cls,
                metrics_fn,
                wandb_config,
                train_config,
                model_config,
                checkpoint
            )
        else:
            path, perf = train_torch(
                args["seed"],
                trainloader,
                testloader,
                model_cls,
                metrics_fn,
                wandb_config,
                train_config,
                model_config,
                checkpoint
            )
    else:
        if args["lang_model"] or (layer == "lru" and data_config["_name_"] == "listops"):
            path, perf = train_jax_step(
                args["seed"],
                trainloader,
                testloader,
                model_cls,
                metrics_fn,
                wandb_config,
                train_config,
                model_config,
                checkpoint,
                data_config
            )
        else:        
            path, perf = train_jax(
                args["seed"],
                trainloader,
                testloader,
                model_cls,
                metrics_fn,
                wandb_config,
                train_config,
                model_config,
                checkpoint,
                data_config
            )
    
    try:
        if wandb_config is not None:
            wandb.finish()
        return path, perf 
    except:
        return path, perf
    
