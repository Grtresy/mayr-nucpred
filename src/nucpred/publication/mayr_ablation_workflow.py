"""Run one preregistered conditional-N ablation from selection to evaluation.

The workflow preserves the publication phase boundary: nested validation selects
epochs, outer-development refits are frozen, all outer-test predictions are
written without labels, and only then is the complete OOF score set evaluated.
Every underlying stage is idempotent, so interrupted GPU campaigns can resume.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import Any

from nucpred.publication.mayr_n_evaluation import (
    evaluate_frozen_oracle,
    freeze_all,
)
from nucpred.publication.mayr_n_modeling import (
    PublicationModelingError,
    preflight,
    read_config,
    run_inner,
    select_outer_epochs,
)
from nucpred.publication.mayr_n_outer import run_outer_refit


ProgressCallback = Callable[[str], None]


def _print_progress(message: str) -> None:
    print(message, flush=True)


def run_ablation(
    config_path: str | Path,
    *,
    device: str | None = None,
    progress: ProgressCallback = _print_progress,
) -> dict[str, Any]:
    """Execute a frozen ablation protocol, resuming completed jobs safely."""

    config, resolved = read_config(config_path)
    ablation = config.get("ablation")
    if not isinstance(ablation, dict) or not ablation.get("name"):
        raise PublicationModelingError(
            "The ablation workflow requires an explicit ablation contract"
        )
    if ablation.get("outer_test_used_for_selection") is not False:
        raise PublicationModelingError("Ablation may not use outer-test outcomes")
    name = str(ablation["name"])
    outer_folds = range(int(config["outer_fold_count"]))
    inner_folds = range(int(config["inner_fold_count"]))
    seeds = tuple(map(int, config["outer_initialization_seeds"]))

    progress(f"[{name}] preflight")
    preflight_payload = preflight(resolved)
    for outer_fold in outer_folds:
        for inner_fold in inner_folds:
            progress(f"[{name}] inner outer={outer_fold} inner={inner_fold}")
            run_inner(
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                config_path=resolved,
                device=device,
            )
        progress(f"[{name}] select outer={outer_fold}")
        select_outer_epochs(outer_fold=outer_fold, config_path=resolved)

    for outer_fold in outer_folds:
        for seed in seeds:
            progress(f"[{name}] outer-refit outer={outer_fold} seed={seed}")
            run_outer_refit(
                outer_fold=outer_fold,
                initialization_seed=seed,
                config_path=resolved,
                device=device,
            )

    progress(f"[{name}] label-blind score freeze")
    frozen = freeze_all(resolved)
    progress(f"[{name}] post-freeze evaluation")
    evaluation = evaluate_frozen_oracle(resolved)
    return {
        "schema_version": "nucpred.mayr-n-publication-ablation-workflow.v1",
        "status": "complete",
        "ablation": name,
        "config_path": resolved.as_posix(),
        "preflight_contract_sha256": preflight_payload["contract_sha256"],
        "outer_score_freeze_count": len(frozen),
        "scores_frozen_before_label_read": evaluation[
            "scores_frozen_before_label_read"
        ],
        "target_count": evaluation["target_count"],
        "metrics": evaluation["metrics"]["stage_e_c_n3"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    result = run_ablation(args.config, device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
