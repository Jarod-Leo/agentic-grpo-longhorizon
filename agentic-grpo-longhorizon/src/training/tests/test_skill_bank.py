"""CPU-only data isolation and resumable extraction with a fake MiMo client."""

import json
import threading
from copy import deepcopy

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


def make_single_candidate(tmp_path):
    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    selected = output / "extraction_inputs.jsonl"
    first = json.loads(selected.read_text().splitlines()[0])
    first["trajectories"] = first["trajectories"][:1]
    selected.write_text(json.dumps(first) + "\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        extraction_inputs_sha256=file_sha256(selected),
        extraction_requests=1,
        selected_trajectories=1,
    )
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, output / "skills.json"


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
        if (
            task["task_id"] == self.fail_on
            or task["trajectories"][0]["trajectory_id"] == self.fail_on_trajectory
        ):
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
    assert [task["task_id"] for task in client.calls] == [1, 1, 1]
    assert all(len(task["trajectories"]) == 1 for task in client.calls)
    assert len(bank["skills"]) == 2
    assert bank["version"] == "skill-bank-v2"
    assert bank["requested_candidates"] == 6
    assert bank["retained_candidates"] == 2
    assert bank["requested_candidates_per_task"] == {"0": 3, "1": 3, "2": 0}
    assert bank["retained_candidates_per_task"] == {"0": 1, "1": 1, "2": 0}
    assert bank["skills"][0]["source_trajectory_ids"] == [
        "train-0",
        "train-2",
        "train-4",
    ]
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
        extract_skill_bank(
            manifest_path,
            FakeMimo(fail_on_trajectory="train-2"),
            WordTokenizer(),
            bank_path,
        )
    cache_path = bank_path.with_suffix(".responses.jsonl")
    cached_before = [json.loads(line) for line in cache_path.read_text().splitlines()]
    cached_ids = {row["source_trajectory_id"] for row in cached_before}
    assert "train-0" in cached_ids
    assert "train-2" not in cached_ids

    client = FakeMimo()
    bank = extract_skill_bank(manifest_path, client, WordTokenizer(), bank_path)

    assert {task["trajectories"][0]["trajectory_id"] for task in client.calls} == {
        f"train-{i}" for i in range(6)
    } - cached_ids
    assert len(cache_path.read_text().splitlines()) == 6
    assert bank["provenance"]["requested_candidates"] == 6
    assert bank["provenance"]["retained_candidates"] == 2
    assert bank["provenance"]["responses_sha256"] == file_sha256(cache_path)
    assert manifest_path.read_bytes() == original_manifest
    original_bank = bank_path.read_bytes()
    with pytest.raises(FileExistsError):
        extract_skill_bank(manifest_path, FakeMimo(), WordTokenizer(), bank_path)
    assert bank_path.read_bytes() == original_bank


def test_bank_v2_retains_distinct_candidates_and_v1_loads_single_candidate_lists(
    tmp_path,
):
    class DistinctMimo(FakeMimo):
        def complete(self, messages, trajectory_id):
            raw, usage = super().complete(messages, trajectory_id)
            fields = json.loads(raw)
            fields["golden_workflow"] += (
                " "
                + json.loads(messages[1]["content"])["trajectories"][0]["trajectory_id"]
            )
            return json.dumps(fields), usage

    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    bank_path = output / "skills.json"
    client = DistinctMimo()
    bank = extract_skill_bank(
        output / "manifest.json", client, WordTokenizer(), bank_path
    )
    loaded = load_skill_bank(bank_path, file_sha256(bank_path), split)

    assert len(client.calls) == 6
    assert bank["requested_candidates"] == bank["retained_candidates"] == 6
    assert len(loaded[0]) == len(loaded[1]) == 3
    assert len({skill["skill_id"] for skill in bank["skills"]}) == 6
    assert all(len(skill["source_trajectory_ids"]) == 1 for skill in bank["skills"])
    bank.update(version="skill-bank-v1", skills=[loaded[0][0], loaded[1][0]])
    legacy = output / "legacy.json"
    legacy.write_text(json.dumps(bank))
    assert load_skill_bank(legacy, file_sha256(legacy), split) == {
        0: [loaded[0][0]],
        1: [loaded[1][0]],
    }

    bank["version"] = "skill-bank-v2"
    bank["skills"][1]["skill_id"] = bank["skills"][0]["skill_id"]
    legacy.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="globally unique"):
        load_skill_bank(legacy, file_sha256(legacy), split)


def test_response_format_changes_request_cache_identity_and_records_provenance(
    tmp_path,
):
    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    manifest = output / "manifest.json"
    bank_path = output / "skills.json"
    with pytest.raises(RuntimeError, match="Mock network"):
        extract_skill_bank(manifest, FakeMimo(fail_on=1), WordTokenizer(), bank_path)
    cache_path = bank_path.with_suffix(".responses.jsonl")
    previous = [
        json.loads(line)["request_sha256"]
        for line in cache_path.read_text().splitlines()
    ]
    client = FakeMimo()
    client.response_format = {"type": "json_object"}

    bank = extract_skill_bank(manifest, client, WordTokenizer(), bank_path)

    assert len(client.calls) == 6
    cached = [
        json.loads(line)["request_sha256"]
        for line in cache_path.read_text().splitlines()
    ]
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
def test_rejected_response_preserves_raw_diagnostics_and_raises(
    tmp_path, raw, budget, error
):
    class RejectedMimo(FakeMimo):
        def complete(self, messages, trajectory_id):
            super().complete(messages, trajectory_id)
            return raw, None

    manifest, bank_path = make_single_candidate(tmp_path)
    client = RejectedMimo()
    with pytest.raises(ValueError, match=error) as caught:
        extract_skill_bank(manifest, client, WordTokenizer(), bank_path, budget)

    rows = [
        json.loads(line)
        for line in bank_path.with_suffix(".rejected.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(client.calls) == 3
    assert rows[0]["task_id"] == 0
    assert rows[0]["source_trajectory_id"] == "train-0"
    assert len(rows[0]["request_sha256"]) == 64
    assert rows[0]["raw"] == raw
    assert rows[0]["error"] == str(caught.value)
    assert rows[0]["error_type"] == type(caught.value).__name__
    assert [row["attempt"] for row in rows] == [1, 2, 3]
    assert len({row["request_sha256"] for row in rows}) == 1
    assert len({row["attempt_sha256"] for row in rows}) == 3
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
        extract_skill_bank(
            output / "manifest.json", SDKFailureMimo(), WordTokenizer(), bank_path
        )
    assert not bank_path.with_suffix(".rejected.jsonl").exists()


@pytest.mark.parametrize(
    ("bad_fields", "reason"),
    [
        ({"tool_calls": 0.88}, "exactly the three"),
        ({"success_analysis": "word " * 161}, "160 total words"),
        (
            {"golden_workflow": "Use credit_card_7815826 and certificate_7815826."},
            "instance-specific ID",
        ),
        ({"golden_workflow": "Use gift_card_7815826."}, "instance-specific ID"),
        (
            {"success_analysis": "User john_doe_1234 requested help."},
            "instance-specific ID",
        ),
        ({"success_analysis": "Travel on 2026-10-11."}, "calendar date"),
        (
            {"golden_workflow": "Call list_reservations."},
            "unavailable tools: list_reservations",
        ),
    ],
)
def test_content_repair_rewrites_invalid_response_without_silent_edits(
    tmp_path, bad_fields, reason
):
    valid = {
        "success_analysis": "Use get_user_details with the current user_id.",
        "mistake_analysis": "No observed mistake.",
        "golden_workflow": "Follow policy for payment_methods and basic_economy; obtain confirmation.",
    }
    invalid_raw = json.dumps({**valid, **bad_fields})

    class CorrectingMimo(FakeMimo):
        def __init__(self):
            super().__init__()
            self.histories = []

        def complete(self, messages, trajectory_id):
            super().complete(messages, trajectory_id)
            self.histories.append(deepcopy(messages))
            return (
                invalid_raw if len(self.histories) == 1 else json.dumps(valid)
            ), None

    manifest, bank_path = make_single_candidate(tmp_path)
    client = CorrectingMimo()
    bank = extract_skill_bank(
        manifest, client, WordTokenizer(), bank_path, allowed_tools=["get_user_details"]
    )

    assert len(client.calls) == 2
    assert client.histories[1][-2] == {"role": "assistant", "content": invalid_raw}
    assert reason in client.histories[1][-1]["content"]
    assert client.calls[0]["allowed_tools"] == ["get_user_details"]
    assert {key: bank["skills"][0][key] for key in valid} == valid
    assert bank["provenance"]["allowed_tools"] == ["get_user_details"]
    assert bank["provenance"]["api_requests_this_run"] == 2
    assert bank["provenance"]["candidate_response_attempts"] == 2
    rejected = json.loads(bank_path.with_suffix(".rejected.jsonl").read_text())
    cached = json.loads(bank_path.with_suffix(".responses.jsonl").read_text())
    assert rejected["raw"] == invalid_raw
    assert cached["content_attempts"] == 2
    assert (
        cached["request_sha256"]
        == rejected["request_sha256"]
        == rejected["attempt_sha256"]
    )
    assert cached["attempt_sha256"] != cached["request_sha256"]


def test_three_content_failures_preserve_audit_and_successful_retry_resumes(tmp_path):
    class ExtraFieldMimo(FakeMimo):
        def complete(self, messages, trajectory_id):
            raw, usage = super().complete(messages, trajectory_id)
            return json.dumps({**json.loads(raw), "tool_calls": 0.88}), usage

    manifest, bank_path = make_single_candidate(tmp_path)
    client = ExtraFieldMimo()
    with pytest.raises(ValueError, match="exactly the three"):
        extract_skill_bank(manifest, client, WordTokenizer(), bank_path)
    assert len(client.calls) == 3
    rejected_path = bank_path.with_suffix(".rejected.jsonl")
    original_rejections = rejected_path.read_bytes()
    assert not bank_path.exists()
    retry = FakeMimo()

    bank = extract_skill_bank(manifest, retry, WordTokenizer(), bank_path)

    assert len(retry.calls) == 1
    assert bank["requested_candidates"] == 1
    assert bank["provenance"]["api_requests_this_run"] == 1
    assert bank["provenance"]["candidate_response_attempts"] == 4
    assert rejected_path.read_bytes() == original_rejections


def test_concurrent_extraction_keeps_input_order_and_valid_caches_after_failure(
    tmp_path,
):
    class ParallelMimo(FakeMimo):
        def __init__(self, fail=False):
            super().__init__()
            self.fail = fail
            self.release_first = threading.Event()
            self.lock = threading.Lock()
            self.active = self.peak = 0
            self.completed = []

        def complete(self, messages, trajectory_id):
            trace_id = json.loads(messages[1]["content"])["trajectories"][0][
                "trajectory_id"
            ]
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                if trace_id == "train-0":
                    assert self.release_first.wait(5), (
                        "Concurrent worker never released the first request"
                    )
                raw, usage = super().complete(messages, trajectory_id)
                with self.lock:
                    self.completed.append(trace_id)
                if trace_id == "train-2":
                    self.release_first.set()
                    if self.fail:
                        raise RuntimeError("Mock parallel failure")
                fields = json.loads(raw)
                fields["golden_workflow"] += " Evidence " + trace_id
                return json.dumps(fields), usage
            finally:
                with self.lock:
                    self.active -= 1

    source, split = make_source(tmp_path)
    output = tmp_path / "corpus"
    collect_successful_trajectories(source, split, output)
    bank_path = output / "skills.json"
    failed = ParallelMimo(fail=True)
    with pytest.raises(RuntimeError, match="parallel failure"):
        extract_skill_bank(
            output / "manifest.json", failed, WordTokenizer(), bank_path, max_workers=4
        )
    assert 2 <= failed.peak <= 4
    assert failed.completed[0] != "train-0"
    cache_path = bank_path.with_suffix(".responses.jsonl")
    cached_before = [json.loads(line) for line in cache_path.read_text().splitlines()]
    cached_ids = {row["source_trajectory_id"] for row in cached_before}
    assert "train-0" in cached_ids
    assert "train-2" not in cached_ids
    retry = ParallelMimo()
    retry.release_first.set()

    bank = extract_skill_bank(
        output / "manifest.json", retry, WordTokenizer(), bank_path, max_workers=4
    )

    assert {task["trajectories"][0]["trajectory_id"] for task in retry.calls} == {
        f"train-{i}" for i in range(6)
    } - cached_ids
    assert [skill["source_trajectory_ids"] for skill in bank["skills"]] == [
        ["train-0"],
        ["train-2"],
        ["train-4"],
        ["train-1"],
        ["train-3"],
        ["train-5"],
    ]
    assert len(cache_path.read_text().splitlines()) == 6
    assert bank["provenance"]["concurrency"] == 4
    assert bank["provenance"]["max_content_attempts"] == 3
