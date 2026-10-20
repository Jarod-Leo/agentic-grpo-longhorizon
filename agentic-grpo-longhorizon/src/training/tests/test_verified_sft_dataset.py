"""Token-exact loss masking for verified SFT."""

import pandas as pd
import pytest

from src.training.verified_sft_dataset import VerifiedTokenSFTDataset


def make_dataset(input_ids, loss_mask, max_length=6):
    dataset = object.__new__(VerifiedTokenSFTDataset)
    dataset.dataframe = pd.DataFrame([{"input_ids": input_ids, "loss_mask": loss_mask}])
    dataset.max_length = max_length
    return dataset


def test_student_prefix_stays_masked_and_teacher_tokens_are_padded():
    sample = make_dataset([1, 2, 3, 4], [0, 0, 1, 1])[0]
    assert sample["input_ids"].tolist() == [1, 2, 3, 4, 0, 0]
    assert sample["loss_mask"].tolist() == [0, 0, 1, 1, 0, 0]
    assert sample["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0]


def test_sample_without_teacher_target_is_rejected():
    with pytest.raises(ValueError, match="no teacher target"):
        make_dataset([1, 2], [0, 0])[0]
