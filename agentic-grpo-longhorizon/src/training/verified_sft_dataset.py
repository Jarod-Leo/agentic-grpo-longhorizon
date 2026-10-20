"""Token-exact SFT dataset produced by the verified tau-bench collector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from verl.utils.fs import copy_local_path_from_hdfs


class VerifiedTokenSFTDataset(Dataset):
    """Pad pre-tokenized samples while preserving the collector's teacher-only mask."""

    def __init__(self, parquet_files, tokenizer, config=None, max_samples=-1):
        del tokenizer
        config = config or {}
        self.max_length = int(config.get("max_length", 24576))
        self.max_samples = int(max_samples)
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed")
        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(copy_local_path_from_hdfs(path, verbose=True)) for path in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        if 0 < self.max_samples < len(self.dataframe):
            if self.shuffle:
                self.dataframe = self.dataframe.sample(n=self.max_samples, random_state=self.seed)
            else:
                self.dataframe = self.dataframe.iloc[: self.max_samples]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        input_ids = np.asarray(row["input_ids"], dtype=np.int64).tolist()
        loss_mask = np.asarray(row["loss_mask"], dtype=np.int64).tolist()
        if len(input_ids) != len(loss_mask):
            raise ValueError("input_ids and loss_mask lengths differ")
        if not input_ids or len(input_ids) > self.max_length:
            raise ValueError(f"sample length {len(input_ids)} is outside 1..{self.max_length}")
        if any(value not in (0, 1) for value in loss_mask):
            raise ValueError("loss_mask must contain only 0 or 1")
        if not any(loss_mask):
            raise ValueError("verified SFT sample has no teacher target tokens")

        length = len(input_ids)
        padding = self.max_length - length
        input_ids = torch.tensor(input_ids + [0] * padding, dtype=torch.long)
        attention_mask = torch.tensor([1] * length + [0] * padding, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask + [0] * padding, dtype=torch.long)
        position_ids = torch.arange(self.max_length, dtype=torch.long) * attention_mask
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
