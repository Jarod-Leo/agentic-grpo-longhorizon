"""CPU tests for shared successful-terminal audit checks."""

from src.evaluation.terminal_audit import summarize_terminal_behavior


def test_terminal_execution_and_inflight_api_request_are_allowed():
    audit_rows = [
        {"trajectory_id": "ok", "event": "generation", "time": 1.0},
        {
            "trajectory_id": "ok",
            "event": "execution",
            "name": "transfer_to_human_agents",
            "error": False,
            "time": 3.0,
        },
    ]
    api_rows = [
        {
            "trajectory_id": "ok",
            "started_at": 2.0,
            "finished_at": 4.0,
            "success": True,
        }
    ]

    summary = summarize_terminal_behavior(audit_rows, api_rows)

    assert summary["passed"] is True
    assert summary["successful_terminal_transfers"] == 1
    assert summary["post_terminal_api_requests"] == 0


def test_calls_after_successful_terminal_transfer_are_reported():
    terminal = {
        "trajectory_id": "bad",
        "event": "execution",
        "name": "transfer_to_human_agents",
        "error": False,
        "time": 3.0,
    }
    audit_rows = [
        terminal,
        {"trajectory_id": "bad", "event": "generation", "time": 4.0},
        {
            "trajectory_id": "bad",
            "event": "execution",
            "name": "get_user_details",
            "error": False,
            "time": 5.0,
        },
    ]
    api_rows = [
        {"trajectory_id": "bad", "started_at": 6.0, "finished_at": 7.0, "success": True}
    ]

    summary = summarize_terminal_behavior(audit_rows, api_rows)

    assert summary["passed"] is False
    assert summary["post_terminal_generation_events"] == 1
    assert summary["post_terminal_execution_events"] == 1
    assert summary["post_terminal_api_requests"] == 1
    assert summary["violating_trajectory_ids"] == ["bad"]


def test_failed_transfer_is_not_a_terminal_event():
    audit_rows = [
        {
            "trajectory_id": "retry",
            "event": "execution",
            "name": "transfer_to_human_agents",
            "error": True,
            "time": 1.0,
        },
        {"trajectory_id": "retry", "event": "generation", "time": 2.0},
    ]

    summary = summarize_terminal_behavior(audit_rows, [])

    assert summary["passed"] is True
    assert summary["successful_terminal_transfers"] == 0
