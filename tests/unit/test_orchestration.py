"""
tests/unit/test_orchestration.py

Unit tests for the Prefect orchestration layer.
No real Spark, no Prefect server, no network — all external deps are mocked.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Prefect mock helpers ───────────────────────────────────────────────────────


def _make_task_decorator(*args, **kwargs):
    """Return a decorator that wraps each function in a unique MagicMock.

    Each call to @task(...) returns a fresh decorator so every task function
    in pipeline.py gets its own MagicMock (with independent .submit() state).
    """

    def decorator(fn):
        return MagicMock(name=f"task_{fn.__name__}")

    return decorator


def _make_prefect_mods() -> dict:
    """Build sys.modules entries that satisfy all Prefect imports in pipeline.py.

    @flow(name=...)(fn) → fn           (flow body executes as plain Python)
    @task(name=...)(fn) → MagicMock()  (each task gets an independent mock)
    allow_failure(x)    → x            (pass-through; wait_for is handled by mock)
    """
    mock_prefect = MagicMock()
    mock_prefect.flow.return_value = lambda f: f
    mock_prefect.task.side_effect = _make_task_decorator

    mock_futures = MagicMock()
    mock_futures.allow_failure.side_effect = lambda x: x

    return {
        "prefect": mock_prefect,
        "prefect.futures": mock_futures,
        "prefect.deployments": MagicMock(),
        "prefect.schedules": MagicMock(),
    }


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def pipeline_mod():
    """Fresh import of pipeline.py with fake Prefect injected into sys.modules.

    Each test that uses this fixture gets independent task MagicMocks because
    the module is re-imported from scratch.
    """
    mods = _make_prefect_mods()
    sys.modules.pop("src.orchestration.pipeline", None)

    with patch.dict(sys.modules, mods):
        from src.orchestration import pipeline as _pipeline

        yield _pipeline

    sys.modules.pop("src.orchestration.pipeline", None)


@pytest.fixture
def schedule_deploy_result():
    """Run schedules.deploy() with mocked Prefect and return captured mocks.

    Returns (mock_deployments, mock_schedules) whose call history can be
    inspected after deploy() has run.
    """
    mods = _make_prefect_mods()
    mock_deployments = MagicMock()
    mock_schedules = MagicMock()
    mods["prefect.deployments"] = mock_deployments
    mods["prefect.schedules"] = mock_schedules

    for key in ("src.orchestration.pipeline", "src.orchestration.schedules"):
        sys.modules.pop(key, None)

    with patch.dict(sys.modules, mods):
        from src.orchestration.schedules import deploy

        deploy()

    for key in ("src.orchestration.pipeline", "src.orchestration.schedules"):
        sys.modules.pop(key, None)

    return mock_deployments, mock_schedules


# ── Flow structure tests ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestRunPipelineStructure:
    def test_returns_dict_with_required_keys(self, pipeline_mod):
        result = pipeline_mod.run_pipeline()

        assert isinstance(result, dict)
        assert "status" in result
        assert "duration_seconds" in result
        assert "tasks_completed" in result
        assert "tasks_failed" in result

    def test_status_success_when_all_tasks_complete(self, pipeline_mod):
        result = pipeline_mod.run_pipeline()

        assert result["status"] == "success"

    def test_status_partial_when_bluesky_fails(self, pipeline_mod):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("Bluesky API unavailable")
        pipeline_mod.ingest_bluesky_task.submit.return_value = fail_future

        result = pipeline_mod.run_pipeline()

        assert result["status"] == "partial"

    def test_status_failed_when_bronze_to_silver_fails(self, pipeline_mod):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("Spark job crashed")
        pipeline_mod.bronze_to_silver_task.submit.return_value = fail_future

        result = pipeline_mod.run_pipeline()

        assert result["status"] == "failed"

    def test_tasks_completed_contains_all_names_on_success(self, pipeline_mod):
        result = pipeline_mod.run_pipeline()

        assert set(result["tasks_completed"]) == {
            "ingest_static",
            "ingest_bluesky",
            "bronze_to_silver",
            "silver_to_gold",
        }

    def test_tasks_failed_is_empty_on_success(self, pipeline_mod):
        result = pipeline_mod.run_pipeline()

        assert result["tasks_failed"] == []

    def test_tasks_failed_contains_bluesky_on_partial(self, pipeline_mod):
        fail_future = MagicMock()
        fail_future.result.side_effect = RuntimeError("network error")
        pipeline_mod.ingest_bluesky_task.submit.return_value = fail_future

        result = pipeline_mod.run_pipeline()

        assert "ingest_bluesky" in result["tasks_failed"]

    def test_duration_seconds_is_positive_float(self, pipeline_mod):
        result = pipeline_mod.run_pipeline()

        assert isinstance(result["duration_seconds"], float)
        assert result["duration_seconds"] > 0.0


# ── Parameter behaviour tests ──────────────────────────────────────────────────


@pytest.mark.unit
class TestRunPipelineParameters:
    def test_run_ingestion_false_skips_static_task(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_ingestion=False)

        pipeline_mod.ingest_static_task.submit.assert_not_called()

    def test_run_ingestion_false_skips_bluesky_task(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_ingestion=False)

        pipeline_mod.ingest_bluesky_task.submit.assert_not_called()

    def test_run_processing_false_skips_bronze_to_silver(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_processing=False)

        pipeline_mod.bronze_to_silver_task.submit.assert_not_called()

    def test_run_processing_false_skips_silver_to_gold(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_processing=False)

        pipeline_mod.silver_to_gold_task.submit.assert_not_called()

    def test_run_bluesky_false_skips_only_bluesky(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_bluesky=False)

        pipeline_mod.ingest_bluesky_task.submit.assert_not_called()
        pipeline_mod.ingest_static_task.submit.assert_called_once()

    def test_run_bluesky_false_does_not_skip_processing(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_bluesky=False)

        pipeline_mod.bronze_to_silver_task.submit.assert_called_once()
        pipeline_mod.silver_to_gold_task.submit.assert_called_once()

    def test_limit_bluesky_passed_to_task(self, pipeline_mod):
        pipeline_mod.run_pipeline(limit_bluesky=50)

        pipeline_mod.ingest_bluesky_task.submit.assert_called_once_with(limit=50)


# ── Schedules tests ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSchedulesDeploy:
    def test_deploy_creates_two_deployments(self, schedule_deploy_result):
        mock_deployments, _ = schedule_deploy_result

        assert mock_deployments.Deployment.build_from_flow.call_count == 2

    def test_deploy_calls_apply_on_both_deployments(self, schedule_deploy_result):
        mock_deployments, _ = schedule_deploy_result

        assert mock_deployments.Deployment.build_from_flow.return_value.apply.call_count == 2

    def test_daily_pipeline_has_cron_schedule(self, schedule_deploy_result):
        _, mock_schedules = schedule_deploy_result

        mock_schedules.CronSchedule.assert_called_once_with("0 2 * * *")

    def test_manual_processing_has_no_schedule(self, schedule_deploy_result):
        mock_deployments, _ = schedule_deploy_result

        calls = mock_deployments.Deployment.build_from_flow.call_args_list
        manual_call = next(
            c for c in calls if c.kwargs.get("name") == "manual-processing-only"
        )
        assert "schedule" not in manual_call.kwargs

    def test_daily_pipeline_name_is_correct(self, schedule_deploy_result):
        mock_deployments, _ = schedule_deploy_result

        calls = mock_deployments.Deployment.build_from_flow.call_args_list
        names = [c.kwargs.get("name") for c in calls]
        assert "daily-full-pipeline" in names

    def test_manual_deployment_name_is_correct(self, schedule_deploy_result):
        mock_deployments, _ = schedule_deploy_result

        calls = mock_deployments.Deployment.build_from_flow.call_args_list
        names = [c.kwargs.get("name") for c in calls]
        assert "manual-processing-only" in names
