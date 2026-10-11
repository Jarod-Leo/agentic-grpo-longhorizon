"""CPU tests for task-local, transactional UCB skill selection."""

import json
import math

import pytest
from src.training.skill_selector import TaskSkillSelector


def make_selector(**kwargs):
    bank = {
        task_id: [{"task_id": task_id, "skill_id": f"task-{task_id}-{suffix}"} for suffix in ("a", "b")]
        for task_id in (0, 1)
    }
    return TaskSkillSelector(bank, "bank-sha", "split-sha", **kwargs)


def test_ucb_tries_untried_skills_then_uses_formula_and_one_group_count_per_task():
    selector = make_selector()
    selected = selector.select([0, 0, 1], step=0)
    assert selected[0]["skill_id"] == "task-0-a"
    assert selected[1]["skill_id"] == "task-1-a"
    audit = selector.commit({0: [1.0, 0.0, 1.0], 1: [0.0, 1.0]}, step=0)
    assert audit[0]["score_before"] is None
    assert audit[0]["count_before"] == 0
    assert audit[0]["count_after"] == 1
    assert audit[0]["reward_count"] == 3
    assert audit[0]["sum_reward_after"] == pytest.approx(2 / 3)
    assert selector.state_dict()["statistics"]["0"]["task-0-a"]["n"] == 1

    selected = selector.select([0, 1], step=1)
    assert selected[0]["skill_id"] == "task-0-b"
    assert selected[1]["skill_id"] == "task-1-b"
    selector.commit({0: [0.0, 0.0], 1: [1.0]}, step=1)
    assert selector.select([0], step=2)[0]["skill_id"] == "task-0-a"
    audit = selector.commit({0: [1.0]}, step=2)
    assert audit[0]["score_before"] == pytest.approx(2 / 3 + math.sqrt(2 * math.log(2)))
    assert audit[0]["mean_reward_before"] == pytest.approx(2 / 3)
    assert audit[0]["task_count_before"] == 2
    assert audit[0]["count_after"] == 2
    assert audit[0]["task_count_after"] == 3


@pytest.mark.parametrize(("coefficient", "expected"), [(0.0, "task-0-a"), (math.sqrt(2), "task-0-b")])
def test_ucb_balances_reward_and_exploration_without_cross_task_counts(coefficient, expected):
    selector = make_selector(exploration_coef=coefficient)
    state = selector.state_dict()
    state["last_step"] = 20
    state["statistics"]["0"] = {
        "task-0-a": {"n": 10, "sum_reward": 10.0},
        "task-0-b": {"n": 1, "sum_reward": 0.0},
    }
    selector.load_state_dict(state)
    selected = selector.select([0, 1], step=21)
    assert selected[0]["skill_id"] == expected
    assert selected[1]["skill_id"] == "task-1-a"
    audit = selector.commit({0: [0.0], 1: [1.0]}, step=21)
    count = 10 if expected == "task-0-a" else 1
    reward = 1.0 if expected == "task-0-a" else 0.0
    assert audit[0]["score_before"] == pytest.approx(reward + coefficient * math.sqrt(math.log(11) / count))
    assert audit[1]["task_count_before"] == 0
    assert audit[1]["count_after"] == 1


def test_ucb_ties_use_bank_order_and_first_mode_always_uses_first_candidate():
    selector = make_selector()
    for step in (0, 1):
        selector.select([0], step)
        selector.commit({0: [1.0]}, step)
    assert selector.select([0], 2)[0]["skill_id"] == "task-0-a"

    fixed = make_selector(mode="first")
    for step in (0, 1, 2):
        assert fixed.select([0], step)[0]["skill_id"] == "task-0-a"
        fixed.commit({0: [0.0]}, step)
    statistics = fixed.state_dict()["statistics"]["0"]
    assert statistics["task-0-a"]["n"] == 3
    assert statistics["task-0-b"]["n"] == 0


def test_pending_retries_and_failed_commit_never_change_statistics():
    selector = make_selector()
    selected = selector.select([0, 1], 1)
    assert selector.select([1, 0, 0], 1) == selected
    with pytest.raises(ValueError, match="pending"):
        selector.select([0, 1], 2)
    with pytest.raises(ValueError, match="pending"):
        selector.select([0], 1)
    with pytest.raises(ValueError, match="pending"):
        selector.state_dict()
    with pytest.raises(ValueError, match="step"):
        selector.commit({0: [1.0], 1: [1.0]}, 2)
    with pytest.raises(ValueError, match="task set"):
        selector.commit({0: [1.0]}, 1)
    with pytest.raises(ValueError, match="binary"):
        selector.commit({0: [1.0], 1: [0.4]}, 1)
    audit = selector.commit({0: [1.0], 1: [0.0]}, 1)
    assert [record["count_after"] for record in audit] == [1, 1]
    assert selector.last_step == 1
    with pytest.raises(ValueError, match="pending"):
        selector.commit({0: [1.0], 1: [0.0]}, 1)
    with pytest.raises(ValueError, match="last committed"):
        selector.select([0], 1)


def test_checkpoint_roundtrip_restores_next_selection_and_audit():
    original = make_selector()
    for step, rewards in ((0, [1.0, 1.0]), (1, [0.0, 1.0])):
        original.select([0, 1], step)
        original.commit({0: rewards, 1: [1.0]}, step)
    state = json.loads(json.dumps(original.state_dict(), allow_nan=False))
    restored = make_selector()
    restored.load_state_dict(state)
    assert restored.last_step == 1
    assert restored.select([0, 1], 2) == original.select([0, 1], 2)
    assert restored.commit({0: [0.0], 1: [1.0]}, 2) == original.commit({0: [0.0], 1: [1.0]}, 2)
    assert restored.state_dict() == original.state_dict()


@pytest.mark.parametrize("field", ["bank_sha256", "split_sha256", "mode", "exploration_coef"])
def test_restore_rejects_different_bank_split_or_configuration(field):
    selector = make_selector()
    state = selector.state_dict()
    state[field] = 0.0 if field == "exploration_coef" else "different"
    with pytest.raises(ValueError, match=field):
        selector.load_state_dict(state)


def test_uncovered_tasks_and_empty_batches_commit_step_without_candidate_statistics():
    selector = make_selector()
    assert selector.select([99, 99], 5) == {99: None}
    audit = selector.commit({99: [0.0, 1.0]}, 5)
    assert audit[0]["skill_id"] is None
    assert audit[0]["reward_mean"] == 0.5
    assert audit[0]["count_after"] == audit[0]["task_count_after"] == 0
    assert selector.last_step == 5
    assert "99" not in selector.state_dict()["statistics"]
    assert selector.select([], 6) == {}
    assert selector.commit({}, 6) == []
    assert selector.last_step == 6


def test_selector_rejects_cross_task_skills_and_duplicate_global_ids():
    with pytest.raises(ValueError, match="belong"):
        TaskSkillSelector({0: [{"task_id": 1, "skill_id": "wrong-task"}]}, "bank-sha", "split-sha")
    with pytest.raises(ValueError, match="globally unique"):
        TaskSkillSelector(
            {0: [{"task_id": 0, "skill_id": "same"}], 1: [{"task_id": 1, "skill_id": "same"}]},
            "bank-sha",
            "split-sha",
        )
