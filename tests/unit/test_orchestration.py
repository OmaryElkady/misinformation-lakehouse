"""
tests/unit/test_orchestration.py

Unit tests for the Prefect orchestration layer.
No real Spark, no Prefect server, no network — all external deps are mocked.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Prefect mock helpers ───────────────────────────────────────────────────────


def _make_task_decorator(*args, **kwargs):
    """Return a decorator that wraps each function in a unique MagicMock.

    Each @task(...) call gets its own decorator so every task in pipeline.py
    gets an independent MagicMock with isolated .submit() state.
    """

    def decorator(fn):
        return MagicMock(name=f"task_{fn.__name__}")

    return decorator


def _make_flow_decorator(*args, **kwargs):
    """Return a decorator that wraps the flow function in a callable object.

    The object:
    - is callable (runs the original flow body)
    - has .deploy() as an AsyncMock (for schedules tests)
    """

    def decorator(fn):
        mock_deploy = AsyncMock()

        class _MockFlow:
            def __call__(self, *a, **kw):
                return fn(*a, **kw)

        obj = _MockFlow()
        obj.deploy = mock_deploy
        return obj

    return decorator


def _make_prefect_mods() -> dict:
    """Build sys.modules entries that satisfy all Prefect imports in pipeline.py."""
    mock_prefect = MagicMock()
    mock_prefect.flow.side_effect = _make_flow_decorator
    mock_prefect.task.side_effect = _make_task_decorator

    return {"prefect": mock_prefect}


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
def schedule_deploy_mock():
    """Run schedules.deploy() and return a namespace with to_deployment and apply mocks.

    schedules.py calls:
      deployment = await run_pipeline.to_deployment(name=..., cron=..., parameters=...)
      await deployment.apply()

    The returned namespace exposes:
      .to_deployment  — AsyncMock recording every to_deployment() call and its kwargs
      .apply          — AsyncMock recording every apply() call
    """
    mods = _make_prefect_mods()

    for key in ("src.orchestration.pipeline", "src.orchestration.schedules"):
        sys.modules.pop(key, None)

    captured: dict = {}

    original_flow_side_effect = mods["prefect"].flow.side_effect

    def capturing_flow_decorator(*args, **kwargs):
        decorator = original_flow_side_effect(*args, **kwargs)

        def capturing_decorator(fn):
            obj = decorator(fn)
            apply_mock = AsyncMock()
            deployment_mock = MagicMock()
            deployment_mock.apply = apply_mock
            to_deployment_mock = AsyncMock(return_value=deployment_mock)
            obj.to_deployment = to_deployment_mock
            captured["to_deployment_mock"] = to_deployment_mock
            captured["apply_mock"] = apply_mock
            return obj

        return capturing_decorator

    mods["prefect"].flow.side_effect = capturing_flow_decorator

    with patch.dict(sys.modules, mods):
        from src.orchestration.schedules import deploy

        deploy()

    for key in ("src.orchestration.pipeline", "src.orchestration.schedules"):
        sys.modules.pop(key, None)

    return types.SimpleNamespace(
        to_deployment=captured["to_deployment_mock"],
        apply=captured["apply_mock"],
    )


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
        pipeline_mod.bronze_to_silver_task.side_effect = RuntimeError("Spark job crashed")

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

    def test_silver_to_gold_skipped_when_bronze_to_silver_fails(self, pipeline_mod):
        pipeline_mod.bronze_to_silver_task.side_effect = RuntimeError("b2s error")

        pipeline_mod.run_pipeline()

        pipeline_mod.silver_to_gold_task.assert_not_called()

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

        pipeline_mod.bronze_to_silver_task.assert_not_called()

    def test_run_processing_false_skips_silver_to_gold(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_processing=False)

        pipeline_mod.silver_to_gold_task.assert_not_called()

    def test_run_bluesky_false_skips_only_bluesky(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_bluesky=False)

        pipeline_mod.ingest_bluesky_task.submit.assert_not_called()
        pipeline_mod.ingest_static_task.submit.assert_called_once()

    def test_run_bluesky_false_does_not_skip_processing(self, pipeline_mod):
        pipeline_mod.run_pipeline(run_bluesky=False)

        pipeline_mod.bronze_to_silver_task.assert_called_once()
        pipeline_mod.silver_to_gold_task.assert_called_once()

    def test_limit_bluesky_passed_to_task(self, pipeline_mod):
        pipeline_mod.run_pipeline(limit_bluesky=50)

        pipeline_mod.ingest_bluesky_task.submit.assert_called_once_with(limit=50)


# ── Schedules tests ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSchedulesDeploy:
    def test_deploy_creates_two_deployments(self, schedule_deploy_mock):
        assert schedule_deploy_mock.to_deployment.call_count == 2

    def test_deploy_calls_apply_on_both_deployments(self, schedule_deploy_mock):
        assert schedule_deploy_mock.apply.call_count == 2

    def test_daily_pipeline_has_cron_schedule(self, schedule_deploy_mock):
        calls = schedule_deploy_mock.to_deployment.call_args_list
        daily_call = next(c for c in calls if c.kwargs.get("name") == "daily-full-pipeline")
        assert daily_call.kwargs.get("cron") == "0 2 * * *"

    def test_manual_processing_has_no_schedule(self, schedule_deploy_mock):
        calls = schedule_deploy_mock.to_deployment.call_args_list
        manual_call = next(c for c in calls if c.kwargs.get("name") == "manual-processing-only")
        assert "cron" not in manual_call.kwargs

    def test_daily_pipeline_name_is_correct(self, schedule_deploy_mock):
        names = [c.kwargs.get("name") for c in schedule_deploy_mock.to_deployment.call_args_list]
        assert "daily-full-pipeline" in names

    def test_manual_deployment_name_is_correct(self, schedule_deploy_mock):
        names = [c.kwargs.get("name") for c in schedule_deploy_mock.to_deployment.call_args_list]
        assert "manual-processing-only" in names
