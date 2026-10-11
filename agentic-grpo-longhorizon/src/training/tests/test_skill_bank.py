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
    (source / "eval_trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
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

    def __init__(self, fail_on=None, fail_on_trajectory=None):
        self.calls = []
        self.fail_on = fail_on
        self.fail_on_trajectory = fail_on_trajectory

    def complete(self, messages, trajectory_id):
        task = json.loads(messages[1]["content"])
        self.calls.append(task)
        if task["task_id"] == self.fail_on or task["trajectories"][0]["trajectory_id"] == self.fail_on_trajectory:
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
    assert result["version"] == "skill-corpus-v2"
    assert result["extraction_requests"] == 4
    assert result["extraction_task_groups"] == 2
    assert result["candidate_unit"] == "trajectory"
    assert result["missing_skill_task_ids"] == [2]
    assert result["selected_trajectories"] == 4
    all_rows = [json.loads(line) for line in (output / "successful_train_trajectories.jsonl").read_text().splitlines()]
    assert len(all_rows) == 6
    assert all(r["score"] == 1 and r["validate"] is False and r["task_id"] in [0, 1] for r in all_rows)
    selected = [json.loads(line) for line in (output / "extraction_inputs.jsonl").read_text().splitlines()]
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
    assert [task["task_id"] for task in client.calls] == [1, 1, 1]
    assert all(len(task["trajectories"]) == 1 for task in client.calls)
    assert len(bank["skills"]) == 2
    assert bank["version"] == "skill-bank-v2"
    assert bank["requested_candidates"] == 6
    assert bank["retained_candidates"] == 2
    assert bank["requested_candidates_per_task"] == {"0": 3, "1": 3, "2": 0}
    assert bank["retained_candidates_per_task"] == {"0": 1, "1": 1, "2": 0}
    assert bank["skills"][0]["source_trajectory_ids"] == ["train-0", "train-2", "train-4"]
    assert len(bank["skills"][0]["extraction_request_sha256s"]) == 3
    loaded = load_skill_bank(bank_path, file_sha256(bank_path), split)
    assert set(loaded) == {0, 1}
    assert len(loaded[0]) == 1
    assert loaded[0][0]["task_id"] == 0
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


def test_extract_resumes_each_request_within_a_task_and_keeps_legacy_corpus(tmp_path):
    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(version="skill-corpus-v1", extraction_requests=2)
    manifest.pop("candidate_unit")
    manifest_path.write_text(json.dumps(manifest))
    original_manifest = manifest_path.read_bytes()
    bank_path = output / "skills-v2.json"
    with pytest.raises(RuntimeError, match="Mock network"):
        extract_skill_bank(manifest_path, FakeMimo(fail_on_trajectory="train-2"), WordTokenizer(), bank_path)
    cache_path = bank_path.with_suffix(".responses.jsonl")
    assert len(cache_path.read_text().splitlines()) == 1

    client = FakeMimo()
    bank = extract_skill_bank(manifest_path, client, WordTokenizer(), bank_path)

    assert len(client.calls) == 5
    assert len(cache_path.read_text().splitlines()) == 6
    assert bank["provenance"]["requested_candidates"] == 6
    assert bank["provenance"]["retained_candidates"] == 2
    assert bank["provenance"]["responses_sha256"] == file_sha256(cache_path)
    assert manifest_path.read_bytes() == original_manifest
    original_bank = bank_path.read_bytes()
    with pytest.raises(FileExistsError):
        extract_skill_bank(manifest_path, FakeMimo(), WordTokenizer(), bank_path)
    assert bank_path.read_bytes() == original_bank


def test_bank_v2_retains_distinct_candidates_and_v1_loads_single_candidate_lists(tmp_path):
    class DistinctMimo(FakeMimo):
        def complete(self, messages, trajectory_id):
            raw, usage = super().complete(messages, trajectory_id)
            fields = json.loads(raw)
            fields["golden_workflow"] += " " + json.loads(messages[1]["content"])["trajectories"][0]["trajectory_id"]
            return json.dumps(fields), usage

    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    bank_path = output / "skills.json"
    client = DistinctMimo()
    bank = extract_skill_bank(output / "manifest.json", client, WordTokenizer(), bank_path)
    loaded = load_skill_bank(bank_path, file_sha256(bank_path), split)

    assert len(client.calls) == 6
    assert bank["requested_candidates"] == bank["retained_candidates"] == 6
    assert len(loaded[0]) == len(loaded[1]) == 3
    assert len({skill["skill_id"] for skill in bank["skills"]}) == 6
    assert all(len(skill["source_trajectory_ids"]) == 1 for skill in bank["skills"])
    bank.update(version="skill-bank-v1", skills=[loaded[0][0], loaded[1][0]])
    legacy = output / "legacy.json"
    legacy.write_text(json.dumps(bank))
    assert load_skill_bank(legacy, file_sha256(legacy), split) == {0: [loaded[0][0]], 1: [loaded[1][0]]}

    bank["version"] = "skill-bank-v2"
    bank["skills"][1]["skill_id"] = bank["skills"][0]["skill_id"]
    legacy.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="globally unique"):
        load_skill_bank(legacy, file_sha256(legacy), split)


def test_response_format_changes_request_cache_identity_and_records_provenance(tmp_path):
    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    manifest = output / "manifest.json"
    bank_path = output / "skills.json"
    with pytest.raises(RuntimeError, match="Mock network"):
        extract_skill_bank(manifest, FakeMimo(fail_on=1), WordTokenizer(), bank_path)
    cache_path = bank_path.with_suffix(".responses.jsonl")
    previous = [json.loads(line)["request_sha256"] for line in cache_path.read_text().splitlines()]
    client = FakeMimo()
    client.response_format = {"type": "json_object"}

    bank = extract_skill_bank(manifest, client, WordTokenizer(), bank_path)

    assert len(client.calls) == 6
    cached = [json.loads(line)["request_sha256"] for line in cache_path.read_text().splitlines()]
    assert cached[:3] == previous
    assert set(previous).isdisjoint(cached[3:])
    assert bank["provenance"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("raw", "budget", "error"),
    [
        ('```json\n{"success_analysis": "Verify."}\n```', 512, "Expecting value"),
        ("{}", 512, "exactly the three skill fields"),
        (
            '{"success_analysis": "Verify.", "mistake_analysis": "None.", "golden_workflow": "Read and confirm."}',
            1,
            "exceeds",
        ),
    ],
)
def test_rejected_response_preserves_raw_diagnostics_and_raises(tmp_path, raw, budget, error):
    class RejectedMimo(FakeMimo):
        def complete(self, messages, trajectory_id):
            super().complete(messages, trajectory_id)
            return raw, None

    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    bank_path = output / "skills.json"
    with pytest.raises(ValueError, match=error) as caught:
        extract_skill_bank(output / "manifest.json", RejectedMimo(), WordTokenizer(), bank_path, budget)

    rows = [json.loads(line) for line in bank_path.with_suffix(".rejected.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["task_id"] == 0
    assert rows[0]["source_trajectory_id"] == "train-0"
    assert len(rows[0]["request_sha256"]) == 64
    assert rows[0]["raw"] == raw
    assert rows[0]["error"] == str(caught.value)
    assert rows[0]["error_type"] == type(caught.value).__name__
    assert not bank_path.exists()
    assert not bank_path.with_suffix(".responses.jsonl").exists()


def test_sdk_exceptions_are_not_written_to_rejected_responses(tmp_path):
    class SDKFailureMimo(FakeMimo):
        def complete(self, messages, trajectory_id):
            raise ValueError("private-provider-error api-key=DO_NOT_LOG")

    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    bank_path = output / "skills.json"
    with pytest.raises(ValueError):
        extract_skill_bank(output / "manifest.json", SDKFailureMimo(), WordTokenizer(), bank_path)
    assert not bank_path.with_suffix(".rejected.jsonl").exists()
