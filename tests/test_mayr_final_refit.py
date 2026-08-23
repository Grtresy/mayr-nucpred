from __future__ import annotations

import math

import pandas as pd
import pytest

from nucpred.publication import mayr_final_refit as final_refit
from nucpred.publication import mayr_site_publication as publication


def test_final_epoch_rule_uses_upper_median() -> None:
    assert final_refit._upper_median([80, 100, 76, 76, 100]) == 80
    assert final_refit._upper_median([154, 144, 138, 122]) == 144


def test_final_epoch_rule_rejects_empty_input() -> None:
    with pytest.raises(publication.PublicationSiteError, match="no epochs"):
        final_refit._upper_median([])


def test_region_key_normalizes_missing_gates_without_changing_grid_identity() -> None:
    row = {
        "arm": "region_structural_residual",
        "minimum_samples_leaf": 2,
        "residual_weight": 1.5,
        "maximum_base_margin": math.nan,
        "top_k": None,
    }

    assert final_refit._region_key(row) == (
        "region_structural_residual",
        2,
        1.5,
        None,
        None,
    )


def test_publication_config_requires_refit_only_after_outer_evaluation() -> None:
    config, _ = publication.read_config()

    assert config["phase_separation"]["final_refit_after_outer_evaluation"] is True
    assert (
        config["phase_separation"]["external_search_after_final_registry_freeze_only"]
        is True
    )
    assert config["outer_test_used_for_selection"] is False


def test_final_margin_gate_abstains_singleton_without_inventing_margin() -> None:
    frame = pd.DataFrame(
        {
            "candidate_count": [1, 2, 3, 4],
            "top1_margin": [math.nan, 0.25, 1.0, 2.0],
            "site_top1_correct": [True, False, True, True],
        }
    )
    settings = {
        "threshold_grid": [0.0, 0.5, 1.0],
        "minimum_development_precision": 1.0,
        "minimum_accepted_count": 1,
    }

    gate = final_refit._select_final_margin_gate(frame, settings)

    assert gate["selected_threshold"] == 0.5
    assert gate["undefined_singleton_context_count"] == 1
    assert gate["undefined_singleton_policy"] == "abstain_margin_undefined"
    assert gate["selection_uses_outer_oof_labels"] is True
    assert gate["selected_coverage_all_oof_contexts"] == 0.5


def test_final_margin_gate_rejects_nan_for_multi_candidate_context() -> None:
    frame = pd.DataFrame(
        {
            "candidate_count": [2],
            "top1_margin": [math.nan],
            "site_top1_correct": [True],
        }
    )
    settings = {
        "threshold_grid": [0.0],
        "minimum_development_precision": 0.5,
        "minimum_accepted_count": 1,
    }

    with pytest.raises(publication.PublicationSiteError, match="Only singleton"):
        final_refit._select_final_margin_gate(frame, settings)
