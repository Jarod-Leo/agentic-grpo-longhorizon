"""CPU-only data isolation and resumable extraction with a fake MiMo client."""

import json

import pytest
from src.training.skill_bank import (
    collect_successful_trajectories,
    extract_skill_bank,
    file_sha256,
    load_skill_bank,
)


def make_source(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"seen_task_ids": [0, 1, 2], "unseen_task_ids": [3]}))
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "trajectory_id": f"train-{i}",
            "step": i + 1,
            "task_id": i % 2,
            "validate": False,
            "score": 1 if i < 6 else 0,
            "split": "train",
            "messages": [
                {"role": "user", "content": f"training-example-{i}"},
                {"role": "tool", "content": "tool evidence"},
            ],
        }
        for i in range(7)
    ]
    rows += [
        dict(rows[0], trajectory_id="evaluation", validate=True),
        dict(rows[0], trajectory_id="heldout", task_id=3, split="test", validate=True),
    ]
    (source / "eval_trajectories.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    # This intentionally exercises E01's legacy metadata without reward_mode.
    (source / "summary.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "metadata": {
                    "mode": "train",
                    "train_task_ids": [0, 1, 2],
                    "test_task_ids": [3],
                    "split_sha256": file_sha256(split),
                },
                "splits": {"train": {"n": 7, "c": 6}},
            }
        )
    )
    return source, split


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeMimo:
    model = "mock-mimo-v2.5"

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def complete(self, messages, trajectory_id):
        task = json.loads(messages[1]["content"])
        self.calls.append(task)
        if task["task_id"] == self.fail_on:
            raise RuntimeError("Mock network failure")
        return json.dumps(
            {
                "success_analysis": "Verify the current reservation.",
                "mistake_analysis": "No observed mistake.",
                "golden_workflow": "Read the booking, explain details, obtain confirmation, update.",
            }
        ), None


def test_collect_excludes_validation_and_preserves_all_train_successes(tmp_path):
    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    result = collect_successful_trajectories(source, split, output, 2)
    assert result["training_trajectories"] == 7
    assert result["successful_trajectories"] == 6
    assert result["excluded_validation_trajectories"] == 2
    assert result["extraction_requests"] == 2
    assert result["missing_skill_task_ids"] == [2]
    assert result["selected_trajectories"] == 4
    all_rows = [
        json.loads(line)
        for line in (output / "successful_train_trajectories.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(all_rows) == 6
    assert all(
        r["score"] == 1 and r["validate"] is False and r["task_id"] in [0, 1]
        for r in all_rows
    )
    selected = [
        json.loads(line)
        for line in (output / "extraction_inputs.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in selected[0]["trajectories"]] == [1, 5]


@pytest.mark.parametrize("bad", ["test", "duplicate", "nonbinary"])
def test_rejects_contamination_or_ambiguous_training_records(tmp_path, bad):
    source, split = make_source(tmp_path)
    path = source / "eval_trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if bad == "test":
        rows[0].update(task_id=3, split="test")
    elif bad == "duplicate":
        rows[1]["trajectory_id"] = rows[0]["trajectory_id"]
    else:
        rows[0]["score"] = 0.4
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        collect_successful_trajectories(source, split, tmp_path / "corpus")


def test_extract_resumes_valid_cached_tasks_and_checks_hashes(tmp_path):
    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    manifest = output / "manifest.json"
    bank_path = output / "skills.json"
    with pytest.raises(RuntimeError, match="Mock network"):
        extract_skill_bank(manifest, FakeMimo(fail_on=1), WordTokenizer(), bank_path)
    client = FakeMimo()
    bank = extract_skill_bank(manifest, client, WordTokenizer(), bank_path)
    assert [task["task_id"] for task in client.calls] == [1]
    assert len(bank["skills"]) == 2
    assert set(load_skill_bank(bank_path, file_sha256(bank_path), split)) == {0, 1}
    with pytest.raises(ValueError, match="SHA256"):
        load_skill_bank(bank_path, "wrong", split)
    bank["skills"][0]["task_id"] = 3
    bank_path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="train-only"):
        load_skill_bank(bank_path, file_sha256(bank_path), split)


def test_extractor_rejects_overlong_skills_and_modified_input(tmp_path):
    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    with pytest.raises(ValueError, match="exceeds"):
        extract_skill_bank(
            output / "manifest.json",
            FakeMimo(),
            WordTokenizer(),
            output / "skills.json",
            1,
        )
    with (output / "extraction_inputs.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="input hash"):
        extract_skill_bank(
            output / "manifest.json",
            FakeMimo(),
            WordTokenizer(),
            output / "skills.json",
        )
