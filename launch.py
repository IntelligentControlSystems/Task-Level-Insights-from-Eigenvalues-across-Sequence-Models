import argparse
from itertools import product
import torch
import wandb
import yaml
import sys
import copy
import os
from pathlib import Path

from train import train
from analysis import eval_eig

default_data_path = None

def process_sweep_config(sweep):
    sweep_keys = {}
    sweep_params = []
    for lvl in sweep:
        key_list = []
        if type(sweep[lvl]) is list:
            param_list = sweep[lvl]
            assert isinstance(param_list, list), "You can only define lists in sweep configs!"
            sweep_params.append(param_list)
            sweep_keys[lvl] = []
        elif type(sweep[lvl]) is dict:
            for param_name in sweep[lvl]:
                param_list = sweep[lvl][param_name]
                assert isinstance(param_list, list), "You can only define lists in sweep configs!"
                key_list.append(param_name)
                sweep_params.append(param_list)
            sweep_keys[lvl] = key_list
    return sweep_keys, list(product(*sweep_params))

def update_args(args, keys, sweep):
    assert len(sweep) == len(keys), "Corrupted sweep configuration detected! Number of keys must match number of arguments."
    i = 0
    for lvl in keys:
        if keys[lvl]:
            for param_name in keys[lvl]:
                args[lvl][param_name] = sweep[i]
                i += 1
        else:
            args[lvl] = sweep[i]
            i += 1
    return args

def launch():    
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="cifar-10.yaml", help="experiment config file")
    parser.add_argument("--analysis_config", type=str, default="no-analysis", help="analysis config file")
    parser.add_argument("--sweep", type=bool, default=False)
    config = parser.parse_args().config
    analysis_config = parser.parse_args().analysis_config
    do_sweep = parser.parse_args().sweep
    print("\nUsing config {0}".format(config))

    # get GPU info
    if not torch.cuda.is_available():
        raise NotImplementedError("Cannot run on CPU!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_type = torch.cuda.get_device_name(0)
    print("Running on {0}".format(gpu_type))
    
    if do_sweep:
        # load sweep config
        with open("configs/"+config) as stream:
            try:
                sweep_args = yaml.safe_load(stream)            
            except yaml.YAMLError as exc:
                raise RuntimeError(exc)
        
        base_config = sweep_args["base_config"]
        sweep = sweep_args["sweep"]

        # get args
        with open("configs/"+base_config) as stream:
            try:
                args = yaml.safe_load(stream)            
            except yaml.YAMLError as exc:
                raise RuntimeError(exc)
    else:
        # get args
        with open("configs/"+config) as stream:
            try:
                args = yaml.safe_load(stream)            
            except yaml.YAMLError as exc:
                raise RuntimeError(exc)

    args["GPU"] = gpu_type
    
    # get wandb config
    if "wandb" in args:
        wandb_config = args.pop("wandb")
    else:
        wandb_config = None
    
    print("\nCONFIG:")
    print(yaml.dump(args))

    # analysis config
    if analysis_config == "no-analysis":
        do_analysis = False
        # get conf_args
    else:
        do_analysis = True
        with open("configs/"+analysis_config) as stream:
            try:
                conf_args = yaml.safe_load(stream)            
            except yaml.YAMLError as exc:
                raise RuntimeError(exc)
    
    # prepare dataset
    data_config = args["dataset"]
    args["lang_model"] = data_config["name"] in ["WikiText", "MQAR"] 

    # Default data path is environment variable or hippo/data
    global default_data_path
    if (default_data_path := os.getenv("DATA_PATH")) is None:
        if "data_dir" in data_config:
            default_data_path = data_config["data_dir"]
        else:
            default_data_path = Path(__file__).parent.parent.absolute()
            default_data_path = default_data_path / "data"
    else:
        default_data_path = Path(default_data_path).absolute()

    from dataloaders import SequenceDataset

    dataset = SequenceDataset.registry[data_config["_name_"]](**data_config)
    dataset.setup()

    # load metrics
    metrics_fn = dataset.get_metrics(layer=args["model"]["layer"])

    train_config = args["train"]
    if "fixed_size" in data_config:
        train_config["padded"] = not data_config["fixed_size"]
    else:
        train_config["padded"] = False
    model_config = args["model"]

    train_config["train_size"] = len(dataset.dataset_train)
    model_config["seq_len"] = dataset.l_max
    bsz = train_config["batch_size"]
    
    # dataloaders
    trainloader = dataset.train_dataloader(batch_size=bsz, shuffle=True)
    testloader = dataset.test_dataloader(batch_size=bsz, shuffle=False)
    if type(testloader) is dict:
        testloader = testloader[None]

    # analysis dataloader
    if do_analysis == True:
        testloader_analysis = dataset.test_dataloader(batch_size=conf_args["batch_size"], shuffle=False)
        if type(testloader_analysis) is dict:
            testloader_analysis = testloader_analysis[None]

    if do_sweep:
        keys, sweeps = process_sweep_config(sweep)
        print("Found {0} sweep configurations ...".format(len(sweeps)))
        print("sweep configs:\n{0}".format(sweeps))
        for idx, step in enumerate(sweeps):
            print("Training... {0}/{1}".format(idx+1, len(sweeps)))
            sweep_args = copy.deepcopy(args)
            sweep_args = update_args(sweep_args, keys, step)
            print("\nCONFIG:")
            print(yaml.dump(sweep_args))
            path, perf = train(sweep_args, wandb_config, trainloader, testloader, metrics_fn)
            if path != None and do_analysis == True:
                print("Running eigenvalue evaluation")
                eval_eig(sweep_args, conf_args, wandb_config, data_config, testloader_analysis, path, perf)
                print("Finished!")
            print("Done with {0} of {1} configurations.".format(idx+1, len(sweeps)))
    else:
        path, perf = train(args, wandb_config,  trainloader, testloader, metrics_fn)
        if path != None and do_analysis:
                print("Running eigenvalue evaluation")
                eval_eig(args, conf_args, wandb_config, data_config, testloader_analysis, path, perf)
                print("Finished!")
        else:
            print("Path is None, no eval")
    
    sys.exit(0)
    

if __name__ == "__main__":
    launch()
