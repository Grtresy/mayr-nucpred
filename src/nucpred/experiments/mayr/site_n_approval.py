"""Record explicit user authorization after the typed site-N stage gate."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from nucpred.artifacts.catalog import ArtifactCatalog
from nucpred.core.files import atomic_write_json, sha256_file
from nucpred.project import get_project_layout

from .site_n import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    SiteNCampaignError,
    _display_path,
    _read_config,
    _write_manifest,
)
from .site_n_stage_gate import STAGE_GATE_RUN_ID


_LAYOUT = get_project_layout()
ROOT = _LAYOUT.root
APPROVAL_SCHEMA = "nucpred.mayr-site-n-stage-approval.v1"
REQUIRED_STATEMENT = "批准进入下一阶段"
APPROVED_SCOPES = (
    "full_pretraining_47915_three_seeds",
    "controlled_downstream_initialization_matrix",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SiteNCampaignError(f"Expected JSON object: {path}")
    return payload


def _statement_sha256(statement: str) -> str:
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()


def _approval_core(
    *,
    statement: str,
    goal_thread_id: str,
    approval_anchor_commit: str,
    stage_gate_summary: Path,
    stage_gate_report: Path,
    stage_gate_catalog_manifest: Path,
) -> dict[str, object]:
    return {
        "schema_version": APPROVAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "approved",
        "authorized_next_stage": True,
        "authorization_source": "explicit_user_message",
        "approval_statement": statement,
        "approval_statement_sha256": _statement_sha256(statement),
        "goal_thread_id": goal_thread_id,
        "approval_anchor_commit": approval_anchor_commit,
        "approved_scopes": list(APPROVED_SCOPES),
        "stage_gate_run_id": STAGE_GATE_RUN_ID,
        "stage_gate_summary_sha256": sha256_file(stage_gate_summary),
        "stage_gate_report_sha256": sha256_file(stage_gate_report),
        "stage_gate_catalog_manifest_sha256": sha256_file(
            stage_gate_catalog_manifest
        ),
    }


def record_approval(
    *,
    statement: str,
    goal_thread_id: str,
    approval_anchor_commit: str,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    if statement != REQUIRED_STATEMENT:
        raise SiteNCampaignError(
            "Approval statement does not match the explicit user authorization"
        )
    if not goal_thread_id.strip():
        raise SiteNCampaignError("Goal thread ID is required")
    if not approval_anchor_commit.strip():
        raise SiteNCampaignError("Approval anchor commit is required")

    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    catalog = ArtifactCatalog()
    catalog_verification = catalog.verify(STAGE_GATE_RUN_ID)
    if (
        catalog_verification["status"] != "pass"
        or catalog_verification["run_status"] != "complete"
    ):
        raise SiteNCampaignError("Frozen stage-gate run did not verify")

    run_directory = catalog.run_directory(STAGE_GATE_RUN_ID)
    stage_gate_summary = (
        run_directory / "campaign" / "stage_gate" / "summary.json"
    )
    stage_gate_report = run_directory / "report.md"
    stage_gate_catalog_manifest = (
        ROOT / "artifacts" / "catalog" / "runs" / f"{STAGE_GATE_RUN_ID}.json"
    )
    summary = _load_json(stage_gate_summary)
    if (
        summary.get("technical_gate_status") != "pass"
        or summary.get("approval_status") != "awaiting_user_approval"
        or summary.get("authorized_next_stage") is not False
    ):
        raise SiteNCampaignError(
            "Frozen stage gate is not passing and approval-blocked"
        )

    core = _approval_core(
        statement=statement,
        goal_thread_id=goal_thread_id,
        approval_anchor_commit=approval_anchor_commit,
        stage_gate_summary=stage_gate_summary,
        stage_gate_report=stage_gate_report,
        stage_gate_catalog_manifest=stage_gate_catalog_manifest,
    )
    campaign_root = (ROOT / str(config["output_root"])).resolve()
    target = campaign_root / "authorization"
    approval_path = target / "approval.json"
    if approval_path.is_file():
        existing = _load_json(approval_path)
        observed_core = {
            key: existing.get(key)
            for key in core
        }
        if observed_core == core:
            return existing
        raise SiteNCampaignError("Existing approval evidence is stale")
    if target.exists():
        raise SiteNCampaignError("Partial approval evidence directory exists")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".authorization.staging-", dir=target.parent)
    )
    try:
        payload: dict[str, object] = {
            **core,
            "approved_at_utc": datetime.now(UTC).isoformat(),
            "stage_gate_catalog_verification": catalog_verification,
            "stage_gate_summary_path": _display_path(stage_gate_summary),
            "stage_gate_report_path": _display_path(stage_gate_report),
            "base_config_path": _display_path(config_file),
            "base_config_sha256": sha256_file(config_file),
        }
        atomic_write_json(
            staging / "approval.json",
            payload,
            ensure_ascii=False,
        )
        _write_manifest(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def verify_approval(
    approval_path: str | Path,
    *,
    required_statement: str,
    required_goal_thread_id: str,
    stage_gate_run_id: str,
) -> dict[str, object]:
    path = Path(approval_path).resolve()
    payload = _load_json(path)
    required_scopes = set(APPROVED_SCOPES)
    observed_scopes = {
        str(value) for value in payload.get("approved_scopes", [])
    }
    checks = {
        "schema": payload.get("schema_version") == APPROVAL_SCHEMA,
        "status": payload.get("status") == "approved",
        "authorized_next_stage": payload.get("authorized_next_stage") is True,
        "authorization_source": (
            payload.get("authorization_source") == "explicit_user_message"
        ),
        "statement": payload.get("approval_statement") == required_statement,
        "statement_hash": payload.get("approval_statement_sha256")
        == _statement_sha256(required_statement),
        "goal_thread_id": (
            payload.get("goal_thread_id") == required_goal_thread_id
        ),
        "stage_gate_run_id": (
            payload.get("stage_gate_run_id") == stage_gate_run_id
        ),
        "approved_scopes": required_scopes.issubset(observed_scopes),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise SiteNCampaignError(f"Approval evidence failed checks: {failed}")
    return {
        "status": "pass",
        "checks": checks,
        "approval_sha256": sha256_file(path),
        "approval": payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement", required=True)
    parser.add_argument("--goal-thread-id", required=True)
    parser.add_argument("--approval-anchor-commit", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    payload = record_approval(
        statement=arguments.statement,
        goal_thread_id=arguments.goal_thread_id,
        approval_anchor_commit=arguments.approval_anchor_commit,
        config_path=arguments.config,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
