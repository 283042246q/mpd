import pytest

from scripts.isaaclab.summarize_replan_timing import summarize_manifest


def _plan(plan_id, start, handoff, end, submitted, status="superseded"):
    return {
        "id": plan_id,
        "status": status,
        "active_from_s": start,
        "active_until_s": end,
        "phase_timing": {
            "planning_submitted_s": submitted,
            "bridge_start_s": start,
            "handoff_s": handoff,
            "mpd_suffix_s": 10.0,
        },
    }


def test_summary_reports_phase_ratio_and_continuous_switches():
    summary = summarize_manifest(
        {
            "plans": [
                _plan("a", 2.0, 2.2, 8.0, 0.5),
                _plan("b", 8.0, 8.3, 14.0, 6.0, "accepted"),
                {"id": "rejected", "status": "rejected"},
            ],
            "events": [{"type": "handoff"}, {"type": "handoff"}],
        }
    )

    assert summary["executed_plan_count"] == 2
    assert summary["maximum_command_gap_s"] == pytest.approx(0.0)
    assert summary["phase_totals"]["initial_planning_wait_s"] == pytest.approx(1.5)
    assert summary["phase_totals"]["old_continuation_s"] == pytest.approx(2.0)
    assert summary["phase_totals"]["quintic_bridge_s"] == pytest.approx(0.5)
    assert summary["phase_totals"]["latest_mpd_realized_s"] == pytest.approx(9.5)
    assert sum(summary["phase_ratios"].values()) == pytest.approx(1.0)


def test_summary_exposes_command_gap_and_brake():
    summary = summarize_manifest(
        {
            "plans": [
                _plan("a", 1.0, 1.2, 5.0, 0.0),
                _plan("b", 5.4, 5.6, 9.0, 4.0, "accepted"),
            ],
            "events": [{"type": "brake"}],
        }
    )

    assert summary["maximum_command_gap_s"] == pytest.approx(0.4)
    assert summary["brake_event_count"] == 1


def test_summary_skips_accepted_plan_not_active_before_recording_ends():
    pending = _plan("pending", 10.0, 10.2, 20.0, 8.0, "accepted")
    pending.pop("active_from_s")
    pending.pop("active_until_s")

    summary = summarize_manifest(
        {
            "plans": [
                _plan("executed", 1.0, 1.2, 9.0, 0.0, "superseded"),
                pending,
            ],
            "events": [{"type": "handoff"}, {"type": "handoff"}],
        }
    )

    assert summary["executed_plan_count"] == 1
    assert summary["pending_plan_count"] == 1
    assert [plan["id"] for plan in summary["plans"]] == ["executed"]
