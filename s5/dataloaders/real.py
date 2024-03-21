import io
import logging
import numpy as np
import os
import pickle
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import torchtext
import torchvision
from einops.layers.torch import Rearrange, Reduce
from datasets import DatasetDict, Value, load_dataset

from .base import default_data_path, SequenceDataset


class RgrReal(SequenceDataset):
    _name_ = "rgr_real"
    d_output = 1
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 4000,
            # # 'max_vocab': 100, # Full size 98
            # "append_bos": False,
            # "append_eos": True,
            "n_workers": 4,  # For tokenizing only
        }

    @property
    def _cache_dir_name(self):
        # return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"
        return f"l_max-{self.l_max}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset = self.process_dataset()

        dataset.set_format(type="torch", columns=["input_seq_num", "target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"], dataset["val"], dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_seq_num"], data["target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            
            # pad inputs
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=0, batch_first=True
            )
            ys = torch.tensor(ys)
            
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "train.tsv"),
                "val": str(self.data_dir / "val.tsv"),
                "test": str(self.data_dir / "test.tsv"),
            },
            delimiter="\t",
            column_names=["target", "input_seq"],
            keep_in_memory=True,
        )
        
        new_features = dataset["train"].features.copy()
        new_features["target"] = Value("float32")
        dataset = dataset.cast(new_features)

        numericalize = lambda example: {
            "input_seq_num": [float(val) for val in example["input_seq"].split(',')],
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["input_seq"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        self.input_len = len(dataset['val']['input_seq_num'][0])
        
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


class ClfReal(SequenceDataset):
    _name_ = "clf_real"
    d_output = 2
    l_output = 0

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def init_defaults(self):
        return {
            "l_max": 4000,
            # 'max_vocab': 100, # Full size 98
            # "append_bos": False,
            # "append_eos": True,
            "n_workers": 4,  # For tokenizing only
        }

    @property
    def _cache_dir_name(self):
        # return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"
        return f"l_max-{self.l_max}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"""
                    File {str(split_path)} not found.
                    """
                    )
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return

        # [2021-08-18] TD: I ran into RuntimeError: Too many open files.
        # https://github.com/pytorch/pytorch/issues/11201
        torch.multiprocessing.set_sharing_strategy("file_system")

        dataset = self.process_dataset()

        dataset.set_format(type="torch", columns=["input_seq_num", "target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"],
            dataset["val"],
            dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_seq_num"], data["target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            
            # pad inputs
            xs = nn.utils.rnn.pad_sequence(
                xs, padding_value=0, batch_first=True
            )
            ys = torch.tensor(ys)
            
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "train.tsv"),
                "val": str(self.data_dir / "val.tsv"),
                "test": str(self.data_dir / "test.tsv"),
            },
            delimiter="\t",
            column_names=["target", "input_seq"],
            keep_in_memory=True,
        )
        # dataset = dataset.remove_columns(["session", "frame"])
        new_features = dataset["train"].features.copy()
        new_features["target"] = Value("int32")
        dataset = dataset.cast(new_features)

        numericalize = lambda example: {
            "input_seq_num": [float(val) for val in example["input_seq"].split(',')],
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["input_seq"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=max(self.n_workers, 1),
        )

        self.input_len = len(dataset['val']['input_seq_num'][0])
        
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
