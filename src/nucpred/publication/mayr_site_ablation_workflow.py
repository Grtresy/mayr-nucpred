"""Run one matched automatic-site ablation through frozen outer evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from nucpred.publication import mayr_site_evaluation as evaluation
from nucpred.publication import mayr_site_publication as shared
from nucpred.publication import mayr_site_scoring as scoring
from nucpred.publication import mayr_site_training as training


ProgressCallback = Callable[[str], None]


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _valid_summary(
    path: Path,
    *,
    schema: str,
    statuses: Sequence[str],
    required: Mapping[str, object] | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema or payload.get("status") not in statuses:
        raise shared.PublicationSiteError(f"Existing stage is stale: {path}")
    for key, value in (required or {}).items():
        if payload.get(key) != value:
            raise shared.PublicationSiteError(
                f"Existing stage contract changed: {path}"
            )
    return payload


def run_site_ablation(
    config_path: str | Path,
    *,
    progress: ProgressCallback = _print_progress,
) -> dict[str, Any]:
    config, resolved = shared.read_config(config_path)
    name = shared.ablation_name(config)
    if name is None:
        raise shared.PublicationSiteError(
            "Automatic-site ablation workflow requires an ablation config"
        )
    root = shared.project_path(config["output_directory"], label="site output")
    progress(f"[{name}/automatic-site] preflight")
    preflight_path = root / "preflight" / "preflight.json"
    preflight_payload = _valid_summary(
        preflight_path,
        schema="nucpred.mayr-n-publication-site-preflight.v1",
        statuses=("pass",),
    )
    if preflight_payload is None:
        preflight_payload = shared.preflight(resolved)

    for outer_fold in range(int(config["outer_fold_count"])):
        for inner_fold in range(int(config["inner_fold_count"])):
            progress(
                f"[{name}/automatic-site] inner outer={outer_fold} inner={inner_fold}"
            )
            path = (
                root
                / "nested_inner"
                / f"outer-{outer_fold}"
                / f"inner-{inner_fold}"
                / "summary.json"
            )
            if (
                _valid_summary(
                    path,
                    schema=training.INNER_SCHEMA,
                    statuses=("pass",),
                    required={"outer_test_target_rows_loaded": 0},
                )
                is None
            ):
                training.run_inner(
                    resolved,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
        progress(f"[{name}/automatic-site] select outer={outer_fold}")
        selection_path = (
            root / "outer_selection" / f"outer-{outer_fold}" / "selection.json"
        )
        if (
            _valid_summary(
                selection_path,
                schema=training.OUTER_SELECTION_SCHEMA,
                statuses=("pass",),
                required={"outer_test_target_rows_loaded": 0},
            )
            is None
        ):
            training.select_outer(resolved, outer_fold=outer_fold)

    for outer_fold in range(int(config["outer_fold_count"])):
        progress(f"[{name}/automatic-site] outer-refit outer={outer_fold}")
        summary_path = root / "outer_refit" / f"outer-{outer_fold}" / "summary.json"
        if (
            _valid_summary(
                summary_path,
                schema=training.OUTER_REFIT_SCHEMA,
                statuses=("pass",),
                required={"outer_test_target_rows_loaded": 0},
            )
            is None
        ):
            training.run_outer_refit(resolved, outer_fold=outer_fold)

    for outer_fold in range(int(config["outer_fold_count"])):
        progress(f"[{name}/automatic-site] label-blind-score outer={outer_fold}")
        summary_path = (
            root / "outer_score_freeze" / f"outer-{outer_fold}" / "summary.json"
        )
        if (
            _valid_summary(
                summary_path,
                schema=scoring.SCORE_SCHEMA,
                statuses=("frozen",),
                required={
                    "site_labels_read_before_score_freeze": False,
                    "N_labels_read_before_score_freeze": False,
                },
            )
            is None
        ):
            scoring.freeze_outer_scores(resolved, outer_fold=outer_fold)

    progress(f"[{name}/automatic-site] post-freeze evaluation")
    evaluation_path = root / "outer_evaluation" / "summary.json"
    evaluation_payload = _valid_summary(
        evaluation_path,
        schema=evaluation.EVALUATION_SCHEMA,
        statuses=("pass",),
        required={"labels_read_only_after_all_score_packages_frozen": True},
    )
    if evaluation_payload is None:
        evaluation_payload = evaluation.evaluate(resolved)
    return {
        "schema_version": "nucpred.mayr-n-publication-site-ablation-workflow.v1",
        "status": "complete",
        "ablation": name,
        "config_path": resolved.relative_to(shared.ROOT).as_posix(),
        "preflight_status": preflight_payload["status"],
        "outer_score_freeze_count": int(config["outer_fold_count"]),
        "scores_frozen_before_labels": evaluation_payload[
            "labels_read_only_after_all_score_packages_frozen"
        ],
        "primary_site_metrics": evaluation_payload["primary_site_metrics"],
        "primary_automatic_site_N_metrics": evaluation_payload[
            "primary_automatic_site_N_metrics"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_site_ablation(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
