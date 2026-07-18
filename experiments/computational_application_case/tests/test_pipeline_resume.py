from __future__ import annotations

import pytest

from experiments.computational_application_case.src.pipeline import CasePipeline


class _FakePath:
    def __init__(self, payload: str = "", exists: bool = True) -> None:
        self.payload = payload
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.payload

    def __repr__(self) -> str:
        return "<fake-path>"


def _minimal_pipeline(fingerprint: str, marker_payload: str) -> CasePipeline:
    pipeline = CasePipeline.__new__(CasePipeline)
    pipeline.force = False
    pipeline.resume = True
    pipeline.skip_figures = False
    pipeline.skip_report = False
    pipeline.run_fingerprint = fingerprint
    pipeline.step_functions = {"repository_audit": lambda: {"ran": True}}
    pipeline._required_artifacts = lambda step: []
    pipeline.marker = lambda step: _FakePath(marker_payload)
    return pipeline


def test_resume_rejects_stale_configuration_or_code_fingerprint() -> None:
    pipeline = _minimal_pipeline("current", '{"run_fingerprint": "stale"}')
    with pytest.raises(RuntimeError, match="changed"):
        pipeline.run(only_step="repository_audit")


def test_resume_rejects_missing_artifact_even_with_matching_marker() -> None:
    pipeline = _minimal_pipeline("current", '{"run_fingerprint": "current"}')
    missing = _FakePath(exists=False)
    pipeline._required_artifacts = lambda step: [missing]
    with pytest.raises(FileNotFoundError, match="artifacts are missing"):
        pipeline.run(only_step="repository_audit")
