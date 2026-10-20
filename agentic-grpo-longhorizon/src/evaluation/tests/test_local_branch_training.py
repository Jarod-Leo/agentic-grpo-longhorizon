"""CPU validation of root-action branch records and pooled signal gates."""

from src.evaluation.local_branching import (
    leave_one_out_advantages,
    summarize_training_branches,
)


def make_record(actions, rewards):
    means = [sum(values) / len(values) for values in rewards]
    advantages = leave_one_out_advantages(means)
    candidates = []
    auxiliary_ids = []
    for candidate_index, (action, values, mean, advantage) in enumerate(
        zip(actions, rewards, means, advantages, strict=True)
    ):
        candidate_id = f"root-branch-c{candidate_index}"
        continuations = []
        for repeat_index, value in enumerate(values):
            branch_id = (
                f"root-branch-c{candidate_index}-m{repeat_index}"
            )
            auxiliary_ids.append(branch_id)
            continuations.append(
                {
                    "repeat_index": repeat_index,
                    "trajectory_id": branch_id,
                    "outcome_reward": value,
                    "suffix_policy_tokens": 0,
                }
            )
        if candidate_index:
            auxiliary_ids.append(candidate_id)
        candidates.append(
            {
                "candidate_index": candidate_index,
                "candidate_request_id": candidate_id,
                "is_root_action": candidate_index == 0,
                "candidate_token_ids": action,
                "candidate_policy_tokens": len(action),
                "mean_reward": mean,
                "local_advantage": advantage,
                "continuations": continuations,
            }
        )
    return {
        "schema_version": "local_branch_root_action_v1",
        "record_type": "group",
        "root_trajectory_id": "root",
        "task_id": 4,
        "split": "train",
        "rollout_n": 0,
        "trigger": "after_tool_error",
        "skipped": False,
        "root_action_span": [3, 3 + len(actions[0])],
        "root_action_token_ids": actions[0],
        "root_local_advantage": advantages[0],
        "candidates": candidates,
        "cost": {
            "root_action_tokens": len(actions[0]),
            "sampled_sibling_actions": len(actions) - 1,
            "sibling_candidate_policy_tokens": sum(
                len(action) for action in actions[1:]
            ),
            "suffix_policy_tokens": 0,
            "branch_user_turns": 0,
            "branch_tool_calls": 0,
            "branch_episodes": sum(len(values) for values in rewards),
            "wall_time_s": 1.0,
        },
        "auxiliary_trajectory_ids": auxiliary_ids,
    }


def summarize(record):
    return summarize_training_branches(
        [record],
        [{"trajectory_id": "root", "task_id": 4, "split": "train"}],
        candidates=4,
        continuations=8,
    )


def test_identical_action_suffix_noise_is_not_action_signal():
    record = make_record(
        [[7, 8]] * 4,
        [[1.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8],
    )

    summary = summarize(record)

    assert summary["engineering_accepted"]
    assert summary["raw_nonzero_advantage_groups"] == 1
    assert summary["nonzero_signal_groups"] == 0
    assert summary["split_half_stable_signal_groups"] == 0


def test_distinct_action_pooled_signal_is_stable_and_audit_ids_are_allowed():
    record = make_record(
        [[7, 8], [9], [10], [11]],
        [[1.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8],
    )

    summary = summarize(record)

    assert summary["engineering_accepted"]
    assert summary["nonzero_signal_groups"] == 1
    assert summary["split_half_stable_signal_groups"] == 1
    assert "root-branch-c1" in summary[
        "allowed_auxiliary_trajectory_ids"
    ]
    assert "root-branch-c0-m0" in summary[
        "allowed_auxiliary_trajectory_ids"
    ]
