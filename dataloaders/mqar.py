"""Wikitext datasets"""
import io
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import torch
from datasets import Dataset, DatasetDict

from dataloaders.base import SequenceDataset
from launch import default_data_path

def multiquery_ar(
        vocab_size: int,
        num_examples: int,
        input_seq_len: int,
        seed: int,
        power_a: float = 0.01,
        num_kv_pairs: int = 8,
        random_non_queries: bool = True,
        include_slices: bool = True,
        **kwargs
) -> Dataset:
    """
    Generates synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    Example:
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)`
        will generate input and label sequences of the form:

                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.

    We include one important note on the power law distribution. In real language data,
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again
    drops the further away from the orginal mention we are. In our synthetic, we can
    control this with the power law parameters `train_power_a` and `test_power_a`.
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology
            paper, large vocabulary sizes (>1k) can be important for highlighting
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In
            In Figure 2 of the Zoology paper, we vary the input sequence length from
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of key-value pairs.
        train_power_a (float, optional): The power for the power law distribution for
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs,
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    
    return Dataset.from_dict({
        "inputs": inputs,
        "labels": labels
    })

class MQAR(SequenceDataset):
    _name_ = "mqar"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "seed": 42,
            "vocab_size": 8_192,
            "num_train_examples": 100_000,
            "num_test_examples": 3_000,
            "input_seq_length": 64,
            "num_kv_pairs": 8,
            "train_power_a": 0.01,
            "test_power_a": 0.01,
            "random_non_queries": True,
        }
    
    @property
    def l_max(self):
        return self.input_seq_length

    def get_metrics(self, layer="s4"):
        if layer in ["mamba", "transformer"]:
            return self.get_metrics_torch()
        else:
            return self.get_metrics_jax()

    def get_metrics_torch(self):
        return lambda y_hat, y, ignore_idx=-100: (y_hat.argmax(dim=-1) == y)[y != ignore_idx].to(float).mean().item()

    def get_metrics_jax(self):
        return lambda y_hat, y, ignore_idx=-100: jnp.mean((y_hat.argmax(axis=-1) == y)[y != ignore_idx].astype(jnp.float32))

    def prepare_data(self):
        self.process_dataset()

    def setup(self, stage=None):
        """If cache_dir is not None, we'll cache the processed dataset there."""
        self.data_dir = self.data_dir or default_data_path / self._name_
        self.cache_dir = self.data_dir / "cache"

        dataset = self.process_dataset()
        print(
            f"MQAR | seq_len {self.input_seq_length} | num_kv_pairs {self.num_kv_pairs} | vocab size {self.vocab_size}"
        )
        dataset.set_format(type="torch", columns=["inputs", "labels"])

        # Create all splits
        self.dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        self.dataset_val = None # don't use validation set

    def _collate_fn(self, batch):
        xs, ys = zip(*[(data["inputs"], data["labels"]) for data in batch])
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys, {"lengths": self.input_seq_length}

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        train_data = multiquery_ar(
            vocab_size=self.vocab_size,
            num_examples=self.num_train_examples,
            input_seq_len=self.input_seq_length,
            seed=self.seed,
            power_a=self.train_power_a,
            num_kv_pairs=self.num_kv_pairs,
            random_non_queries=self.random_non_queries,
        )
        test_data = multiquery_ar(
            vocab_size=self.vocab_size,
            num_examples=self.num_test_examples,
            input_seq_len=self.input_seq_length,
            seed=self.seed,
            power_a=self.test_power_a,
            num_kv_pairs=self.num_kv_pairs,
            random_non_queries=self.random_non_queries,
        )
        dataset = DatasetDict(train=train_data, test=test_data)

        if cache_dir is not None:
            self._save_to_cache(dataset, cache_dir)
        return dataset

    def _save_to_cache(self, dataset, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))

        return dataset

    @property
    def _cache_dir_name(self):
        return f"seq_len-{self.input_seq_length}-num_kv_pairs-{self.num_kv_pairs}-vocab_size-{self.vocab_size}"

