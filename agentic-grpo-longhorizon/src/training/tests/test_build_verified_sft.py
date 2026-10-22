"""Verified SFT filtering keeps collection failures separate from environment results."""

import runpy
from pathlib import Path


builder = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "scripts/train/sft/build_verified_sft.py")
)
select_verified_records = builder["select_verified_records"]


def test_teacher_truncation_is_excluded_and_not_an_environment_failure():
    success = {
        "sft_mode": "teacher",
        "task_id": 0,
        "split": "train",
        "score": 1,
        "terminal_semantics": "first_environment_done_v1",
        "termination": "environment_terminated",
        "contaminated": False,
        "collection_failure": None,
        "teacher_takeover": True,
        "input_ids": [10, 20],
        "loss_mask": [0, 1],
        "loss_start": 1,
        "trajectory_id": "success",
    }
    truncated = {
        **success,
        "score": 0,
        "collection_failure": "teacher_output_truncated",
        "termination": "teacher_output_truncated",
        "trajectory_id": "truncated",
    }

    selected, rejected = select_verified_records(
        [success, truncated], {0}, "teacher"
    )

    assert [row["trajectory_id"] for row in selected] == ["success"]
    assert rejected == {"collection_failure:teacher_output_truncated": 1}
