"""Wikitext datasets"""
import io
import logging
import os
import pickle
from pathlib import Path
import jax.numpy as jnp
import jax

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import DatasetDict, load_dataset

from dataloaders.base import SequenceDataset
from jax_helpers import loss_fn


def get_default_data_path():
    from launch import default_data_path
    return default_data_path

class WikiText(SequenceDataset):
    _name_ = "wikitext"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "version": 2,
            "block_size": 1024,
            "seed": 42,
            "n_workers": 8,  # Only used for tokenizing dataset before caching
        }

    @property
    def n_tokens(self):
        return self.vocab_size
    
    @property
    def l_max(self):
        return self.block_size

    def get_metrics(self, layer="s4"):
        if layer in ["mamba", "transformer"]:
            return self.get_metrics_torch()
        else:
            return self.get_metrics_jax()
    
    def get_metrics_torch(self):
        return lambda y_hat, y: torch.exp(F.cross_entropy(y_hat.reshape(-1, y_hat.size(-1)), y.reshape(-1))).item()

    def get_metrics_jax(self):
        return lambda y_hat, y: jnp.exp(loss_fn(y_hat, y))
    
    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
                load_dataset(self._name_, "{0}-{1}-raw-v1".format(self._name_, self.version), cache_dir=self.data_dir)        
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or get_default_data_path() / self._name_
        self.cache_dir = self.data_dir / "cache"

        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        print(
            f"WikiText-{self.version} | tokenizer {self.tokenizer.name_or_path} | vocab size {len(self.vocab)}"
        )
        dataset.set_format(type="torch", columns=["input_ids", "labels"])

        # Create all splits
        self.dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        self.dataset_val = None # don't use validation set

    def _collate_fn(self, batch):
        xs, ys = zip(*[(data["input_ids"], data["labels"]) for data in batch])
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys, {"lengths": self.block_size}

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset("Salesforce/wikitext", "{0}-{1}-raw-v1".format(self._name_, self.version), cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"]) # remove validation

        # Use the GPT-2 tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        vocab = tokenizer.vocab

        # tokenize
        tokenize = lambda example: tokenizer(example["text"])
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            batched=True,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        # group inputs (ensure equal length)
        def group_inputs(examples):
            # Concatenate all tokenized input_ids
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            # Truncate to a multiple of block_size
            total_length = (total_length // self.block_size) * self.block_size
            # Split into chunks of block_size
            result = {
                k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
                for k, t in concatenated.items()
            }
            # Labels = input_ids for causal language modeling
            result["labels"] = result["input_ids"].copy()
            return result
        
        dataset = dataset.map(
            group_inputs,
            batched=True,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        # shift labels
        def shift(examples):
            result = [x[1:] + [-100] for x in examples["labels"]]
            return {"labels": result}
        
        dataset = dataset.map(
            shift,
            batched=True,
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"version-{self.version}-block_size-{self.block_size}"

