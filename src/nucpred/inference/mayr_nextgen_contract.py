"""Fail-closed contract validation for next-generation Mayr inference."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from rdkit import Chem

from nucpred.core.files import sha256_file
from nucpred.project import get_project_layout


ROOT = get_project_layout().root
REQUEST_SCHEMA_PATH = (
    ROOT / "schemas/inference/mayr-nextgen-request.schema.json"
)
RESPONSE_SCHEMA_PATH = (
    ROOT / "schemas/inference/mayr-nextgen-response.schema.json"
)
REQUEST_SCHEMA_VERSION = "nucpred.mayr-nextgen-inference-request.v1"
RESPONSE_SCHEMA_VERSION = "nucpred.mayr-nextgen-inference-response.v1"
SITE_TYPES = (
    "atom",
    "atom_group",
    "bond",
    "delocalized_region",
    "transferable_h_group",
)
FORBIDDEN_KEYS = frozenset(
    {
        "normalized_relative_score",
        "site_probability_distribution",
        "site_softmax",
        "no_nucleophilic_site",
        "no_nucleophile",
        "target_N",
        "N_target",
        "site_label",
        "validity_label",
    }
)


class MayrInferenceContractError(ValueError):
    """Raised when an inference document violates structure or semantics."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MayrInferenceContractError(f"Cannot read JSON {path}: {exc}") from exc


def _load_schema(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise MayrInferenceContractError(f"Schema is not an object: {path}")
    Draft202012Validator.check_schema(payload)
    return payload


def _format_validation_error(error: Any) -> str:
    location = ".".join(str(value) for value in error.absolute_path)
    prefix = location or "<root>"
    return f"{prefix}: {error.message}"


def _validate_schema(
    payload: object,
    *,
    schema_path: Path,
) -> None:
    schema = _load_schema(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (
            tuple(str(value) for value in error.absolute_path),
            error.message,
        ),
    )
    if errors:
        rendered = "; ".join(_format_validation_error(error) for error in errors)
        raise MayrInferenceContractError(rendered)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MayrInferenceContractError(f"{label} must be an object")
    return value


def _walk_forbidden_keys(value: object, *, path: str = "<root>") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_KEYS:
                raise MayrInferenceContractError(
                    f"{path}.{key_text}: forbidden inference field"
                )
            _walk_forbidden_keys(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden_keys(child, path=f"{path}[{index}]")


def _walk_nonfinite_numbers(value: object, *, path: str = "<root>") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise MayrInferenceContractError(f"{path}: non-finite number is forbidden")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_nonfinite_numbers(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_nonfinite_numbers(child, path=f"{path}[{index}]")


def validate_request(payload: object) -> dict[str, object]:
    """Validate that inference input is target-blind and chemically scoped."""
    _walk_forbidden_keys(payload)
    _walk_nonfinite_numbers(payload)
    _validate_schema(payload, schema_path=REQUEST_SCHEMA_PATH)
    request = _mapping(payload, label="request")
    return {
        "status": "pass",
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_id": str(request["request_id"]),
        "chemical_input_fields": ["smiles", "solvent", "formal_charge"],
        "target_or_site_label_present": False,
        "request_schema_sha256": sha256_file(REQUEST_SCHEMA_PATH),
    }


def _validate_membership(candidate: Mapping[str, Any], *, index: int) -> None:
    site_type = str(candidate["site_type"])
    membership = _mapping(candidate["membership"], label=f"candidates[{index}].membership")
    atoms = [int(value) for value in membership["member_atom_indices"]]
    atomic_numbers = [int(value) for value in membership["member_atomic_numbers"]]
    bonds = [[int(value) for value in pair] for pair in membership["member_bond_pairs"]]
    parents = [
        int(value) for value in membership["hydrogen_parent_atom_indices"]
    ]
    prefix = f"candidates[{index}]"
    if atoms != sorted(atoms) or len(atoms) != len(set(atoms)):
        raise MayrInferenceContractError(
            f"{prefix}: member atom indices must be sorted and unique"
        )
    if len(atomic_numbers) != len(atoms):
        raise MayrInferenceContractError(
            f"{prefix}: atomic-number and atom-index lengths differ"
        )
    normalized_bonds: list[tuple[int, int]] = []
    for pair in bonds:
        left, right = pair
        if left >= right:
            raise MayrInferenceContractError(
                f"{prefix}: bond pairs must be sorted with left < right"
            )
        if left not in atoms or right not in atoms:
            raise MayrInferenceContractError(
                f"{prefix}: bond pair escapes candidate atom membership"
            )
        normalized_bonds.append((left, right))
    if len(normalized_bonds) != len(set(normalized_bonds)):
        raise MayrInferenceContractError(f"{prefix}: duplicate member bond pair")
    if normalized_bonds != sorted(normalized_bonds):
        raise MayrInferenceContractError(f"{prefix}: bond pairs must be sorted")

    if site_type == "atom":
        if len(atoms) != 1 or bonds or parents:
            raise MayrInferenceContractError(
                f"{prefix}: atom candidate must contain one atom and no bonds/parents"
            )
    elif site_type == "bond":
        if len(atoms) != 2 or len(bonds) != 1 or set(bonds[0]) != set(atoms):
            raise MayrInferenceContractError(
                f"{prefix}: bond candidate membership is inconsistent"
            )
        if parents:
            raise MayrInferenceContractError(
                f"{prefix}: non-H candidate cannot expose hydrogen parents"
            )
    elif site_type == "delocalized_region":
        if len(atoms) < 2 or not bonds or parents:
            raise MayrInferenceContractError(
                f"{prefix}: delocalized region requires atoms/bonds and no H parents"
            )
    elif site_type == "atom_group":
        if len(atoms) < 2 or parents:
            raise MayrInferenceContractError(
                f"{prefix}: atom group requires multiple atoms and no H parents"
            )
    elif site_type == "transferable_h_group":
        if any(number != 1 for number in atomic_numbers):
            raise MayrInferenceContractError(
                f"{prefix}: transferable-H group contains a non-H atom"
            )
        if len(parents) != len(atoms):
            raise MayrInferenceContractError(
                f"{prefix}: transferable-H group must map every H to a parent"
            )
        if bonds:
            raise MayrInferenceContractError(
                f"{prefix}: transferable-H membership uses parents, not internal bonds"
            )


def _validate_validity(candidate: Mapping[str, Any], *, index: int) -> None:
    validity = _mapping(candidate["validity"], label=f"candidates[{index}].validity")
    logit = validity["logit_mean"]
    logit_std = validity["logit_std"]
    raw_score = validity["raw_sigmoid_score"]
    probability = validity["absolute_site_probability"]
    probability_status = str(validity["probability_status"])
    reason = validity["probability_unavailable_reason"]
    prefix = f"candidates[{index}].validity"
    if (logit is None) != (raw_score is None) or (logit is None) != (
        logit_std is None
    ):
        raise MayrInferenceContractError(
            f"{prefix}: logit mean/std and raw sigmoid score availability differ"
        )
    if logit is not None:
        expected = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(logit)))))
        if not math.isclose(
            expected,
            float(raw_score),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise MayrInferenceContractError(
                f"{prefix}: raw sigmoid score does not match independent logit"
            )
    if probability_status == "calibrated":
        if probability is None or reason is not None:
            raise MayrInferenceContractError(
                f"{prefix}: calibrated probability is incomplete"
            )
    elif probability is not None or not isinstance(reason, str) or not reason:
        raise MayrInferenceContractError(
            f"{prefix}: unavailable probability must be null with a reason"
        )
    support_status = str(validity["support_status"])
    raw_available = raw_score is not None
    if support_status == "unsupported" and raw_available:
        raise MayrInferenceContractError(
            f"{prefix}: unsupported raw score must be null"
        )
    if support_status != "unsupported" and not raw_available:
        raise MayrInferenceContractError(
            f"{prefix}: supported ranking score is missing"
        )


def _validate_conditional_n(candidate: Mapping[str, Any], *, index: int) -> None:
    output = _mapping(
        candidate["conditional_N"],
        label=f"candidates[{index}].conditional_N",
    )
    available = output["status"] == "available"
    if available != (output["mean"] is not None and output["std"] is not None):
        raise MayrInferenceContractError(
            f"candidates[{index}].conditional_N: status/value mismatch"
        )


def _validate_aggregate(response: Mapping[str, Any]) -> None:
    candidates = [
        _mapping(value, label=f"candidates[{index}]")
        for index, value in enumerate(response["candidates"])
    ]
    aggregate = _mapping(response["aggregate"], label="aggregate")
    observed_types = Counter(str(candidate["site_type"]) for candidate in candidates)
    expected = {
        "returned_candidate_count": len(candidates),
        "raw_score_available_candidate_count": sum(
            candidate["validity"]["raw_sigmoid_score"] is not None
            for candidate in candidates
        ),
        "probability_available_candidate_count": sum(
            candidate["validity"]["absolute_site_probability"] is not None
            for candidate in candidates
        ),
        "conditional_N_available_candidate_count": sum(
            candidate["conditional_N"]["status"] == "available"
            for candidate in candidates
        ),
    }
    for field, value in expected.items():
        if int(aggregate[field]) != int(value):
            raise MayrInferenceContractError(
                f"aggregate.{field}: expected {value}, observed {aggregate[field]}"
            )
    by_type = _mapping(
        aggregate["candidate_count_by_type"],
        label="aggregate.candidate_count_by_type",
    )
    for site_type in SITE_TYPES:
        if int(by_type[site_type]) != observed_types[site_type]:
            raise MayrInferenceContractError(
                f"aggregate.candidate_count_by_type.{site_type}: count mismatch"
            )


def _validate_probability_provenance(response: Mapping[str, Any]) -> None:
    scope = _mapping(response["scope"], label="scope")
    provenance = _mapping(response["provenance"], label="provenance")
    candidates = list(response["candidates"])
    calibrated = [
        candidate
        for candidate in candidates
        if candidate["validity"]["probability_status"] == "calibrated"
    ]
    calibrator = provenance["calibrator_run_id"]
    calibrator_scope = str(scope["calibrator_scope"])
    calibrated_site_types = [str(value) for value in scope["calibrated_site_types"]]
    if calibrated_site_types != [
        site_type for site_type in SITE_TYPES if site_type in calibrated_site_types
    ]:
        raise MayrInferenceContractError(
            "scope.calibrated_site_types must follow canonical site-type order"
        )
    if calibrated and (calibrator is None or calibrator_scope == "unavailable"):
        raise MayrInferenceContractError(
            "calibrated candidate probability lacks calibrator provenance"
        )
    if calibrator_scope == "unavailable":
        if calibrator is not None or calibrated_site_types or calibrated:
            raise MayrInferenceContractError(
                "unavailable calibrator scope must expose no calibrator or types"
            )
    elif calibrator is None:
        raise MayrInferenceContractError(
            "available calibrator scope must name a calibrator run"
        )
    if calibrator_scope == "partial_site_types" and not (
        0 < len(calibrated_site_types) < len(SITE_TYPES)
    ):
        raise MayrInferenceContractError(
            "partial calibrator scope must list one to four site types"
        )
    if calibrator_scope == "all_site_types" and tuple(calibrated_site_types) != SITE_TYPES:
        raise MayrInferenceContractError(
            "all-site calibrator scope must list all five site types"
        )
    for index, candidate in enumerate(candidates):
        site_type = str(candidate["site_type"])
        validity = candidate["validity"]
        probability_status = str(validity["probability_status"])
        if probability_status == "calibrated" and site_type not in calibrated_site_types:
            raise MayrInferenceContractError(
                f"candidates[{index}]: calibrated probability is outside calibrator scope"
            )
        if (
            site_type in calibrated_site_types
            and validity["raw_sigmoid_score"] is not None
            and candidate["applicability"]["status"] in {"supported", "uncertain"}
            and probability_status != "calibrated"
        ):
            raise MayrInferenceContractError(
                f"candidates[{index}]: in-scope usable score was not calibrated"
            )


def _validate_model_provenance(response: Mapping[str, Any]) -> None:
    provenance = _mapping(response["provenance"], label="provenance")
    expected_schema_hash = sha256_file(RESPONSE_SCHEMA_PATH)
    if provenance["contract_schema_sha256"] != expected_schema_hash:
        raise MayrInferenceContractError(
            "provenance.contract_schema_sha256 does not match the locked schema"
        )
    candidates = list(response["candidates"])
    raw_score_available = any(
        candidate["validity"]["raw_sigmoid_score"] is not None
        for candidate in candidates
    )
    conditional_n_available = any(
        candidate["conditional_N"]["status"] == "available"
        for candidate in candidates
    )
    if raw_score_available and provenance["validity_model_run_id"] is None:
        raise MayrInferenceContractError(
            "raw validity score lacks validity-model provenance"
        )
    if conditional_n_available and provenance["N_model_run_id"] is None:
        raise MayrInferenceContractError(
            "conditional N output lacks N-model provenance"
        )


def _validate_molecular_grounding(response: Mapping[str, Any]) -> None:
    if response["status"] == "refused":
        return
    echo = _mapping(response["input"], label="input")
    molecule = Chem.MolFromSmiles(str(echo["smiles"]))
    if molecule is None:
        raise MayrInferenceContractError(
            "non-refused response contains an invalid SMILES"
        )
    observed_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    if observed_charge != int(echo["formal_charge"]):
        raise MayrInferenceContractError(
            "input.formal_charge does not match the parsed molecular graph"
        )
    try:
        all_atom_molecule = Chem.AddHs(molecule)
    except Exception as exc:
        raise MayrInferenceContractError(
            f"cannot construct deterministic all-atom graph: {exc}"
        ) from exc
    atom_count = all_atom_molecule.GetNumAtoms()
    for index, raw_candidate in enumerate(response["candidates"]):
        candidate = _mapping(raw_candidate, label=f"candidates[{index}]")
        membership = _mapping(
            candidate["membership"],
            label=f"candidates[{index}].membership",
        )
        atoms = [int(value) for value in membership["member_atom_indices"]]
        if any(atom_index >= atom_count for atom_index in atoms):
            raise MayrInferenceContractError(
                f"candidates[{index}]: member atom index exceeds all-atom graph"
            )
        observed_atomic_numbers = [
            all_atom_molecule.GetAtomWithIdx(atom_index).GetAtomicNum()
            for atom_index in atoms
        ]
        if observed_atomic_numbers != list(membership["member_atomic_numbers"]):
            raise MayrInferenceContractError(
                f"candidates[{index}]: atomic numbers do not match input graph"
            )
        atom_set = set(atoms)
        observed_bonds = sorted(
            [
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            ]
            for bond in all_atom_molecule.GetBonds()
            if {
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
            }
            <= atom_set
        )
        if observed_bonds != list(membership["member_bond_pairs"]):
            raise MayrInferenceContractError(
                f"candidates[{index}]: internal bonds do not match input graph"
            )
        if candidate["site_type"] == "transferable_h_group":
            observed_parents: list[int] = []
            for atom_index in atoms:
                atom = all_atom_molecule.GetAtomWithIdx(atom_index)
                neighbours = [neighbour.GetIdx() for neighbour in atom.GetNeighbors()]
                if len(neighbours) != 1:
                    raise MayrInferenceContractError(
                        f"candidates[{index}]: H member has no unique parent"
                    )
                observed_parents.append(neighbours[0])
            if observed_parents != list(
                membership["hydrogen_parent_atom_indices"]
            ):
                raise MayrInferenceContractError(
                    f"candidates[{index}]: H-parent mapping does not match input graph"
                )


def _validate_status(response: Mapping[str, Any]) -> None:
    status = str(response["status"])
    feature_status = _mapping(response["feature_status"], label="feature_status")
    applicability = str(feature_status["applicability_status"])
    reasons = list(feature_status["refusal_reasons"])
    candidates = list(response["candidates"])
    globally_unusable = (
        feature_status["structure"] == "failed"
        or feature_status["candidate_generation"] == "failed"
        or feature_status["solvent"] in {"unsupported", "invalid"}
    )
    if globally_unusable and status != "refused":
        raise MayrInferenceContractError(
            "globally unusable structure/candidates/solvent must be refused"
        )
    if applicability in {"out_of_domain", "feature_failure"} and status != "refused":
        raise MayrInferenceContractError(
            "out-of-domain or globally failed input must be refused"
        )
    if status == "refused" and (response["candidates"] or not reasons):
        raise MayrInferenceContractError(
            "refused response must have zero candidates and at least one reason"
        )
    if status == "refused" and applicability not in {
        "out_of_domain",
        "feature_failure",
    }:
        raise MayrInferenceContractError(
            "refusal requires out-of-domain or feature-failure status"
        )
    if status == "ok":
        aggregate = _mapping(response["aggregate"], label="aggregate")
        returned = int(aggregate["returned_candidate_count"])
        if (
            applicability != "in_domain"
            or feature_status["candidate_generation"] != "complete"
            or reasons
            or int(aggregate["raw_score_available_candidate_count"]) != returned
            or int(aggregate["probability_available_candidate_count"]) != returned
            or int(aggregate["conditional_N_available_candidate_count"]) != returned
            or any(
                candidate["applicability"]["status"] != "supported"
                for candidate in candidates
            )
        ):
            raise MayrInferenceContractError(
                "status=ok requires complete supported outputs for every candidate"
            )
    if status == "partial":
        aggregate = _mapping(response["aggregate"], label="aggregate")
        returned = int(aggregate["returned_candidate_count"])
        complete = (
            applicability == "in_domain"
            and feature_status["candidate_generation"] == "complete"
            and not reasons
            and int(aggregate["raw_score_available_candidate_count"]) == returned
            and int(aggregate["probability_available_candidate_count"]) == returned
            and int(aggregate["conditional_N_available_candidate_count"]) == returned
            and all(
                candidate["applicability"]["status"] == "supported"
                for candidate in candidates
            )
        )
        if complete:
            raise MayrInferenceContractError(
                "status=partial is inconsistent with complete supported outputs"
            )


def validate_response(
    payload: object,
    *,
    request: object | None = None,
) -> dict[str, object]:
    """Validate structure, independent-score semantics, and provenance."""
    _walk_forbidden_keys(payload)
    _walk_nonfinite_numbers(payload)
    _validate_schema(payload, schema_path=RESPONSE_SCHEMA_PATH)
    response = _mapping(payload, label="response")
    if request is not None:
        validate_request(request)
        request_mapping = _mapping(request, label="request")
        if str(response["request_id"]) != str(request_mapping["request_id"]):
            raise MayrInferenceContractError("response request_id does not match")
        echo = _mapping(response["input"], label="response.input")
        for field in ("smiles", "solvent", "formal_charge"):
            if echo[field] != request_mapping[field]:
                raise MayrInferenceContractError(
                    f"response input echo changed {field}"
                )
    candidates = [
        _mapping(value, label=f"candidates[{index}]")
        for index, value in enumerate(response["candidates"])
    ]
    ids = [str(candidate["candidate_site_id"]) for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise MayrInferenceContractError("candidate_site_id values are not unique")
    signatures = [
        (
            str(candidate["site_type"]),
            tuple(candidate["membership"]["member_atom_indices"]),
            tuple(
                tuple(pair)
                for pair in candidate["membership"]["member_bond_pairs"]
            ),
        )
        for candidate in candidates
    ]
    if len(signatures) != len(set(signatures)):
        raise MayrInferenceContractError(
            "duplicate candidate type/membership signatures"
        )
    for index, candidate in enumerate(candidates):
        _validate_membership(candidate, index=index)
        _validate_validity(candidate, index=index)
        _validate_conditional_n(candidate, index=index)
    _validate_aggregate(response)
    _validate_probability_provenance(response)
    _validate_model_provenance(response)
    _validate_molecular_grounding(response)
    _validate_status(response)
    calibrated_count = sum(
        candidate["validity"]["probability_status"] == "calibrated"
        for candidate in candidates
    )
    return {
        "status": "pass",
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": str(response["request_id"]),
        "response_status": str(response["status"]),
        "candidate_count": len(candidates),
        "calibrated_probability_count": calibrated_count,
        "candidate_scores_independent": True,
        "candidate_softmax_used": False,
        "no_site_claimed": False,
        "dft_or_cdft_used": False,
        "target_or_site_label_read": False,
        "response_schema_sha256": sha256_file(RESPONSE_SCHEMA_PATH),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", type=Path)
    group.add_argument("--response", type=Path)
    parser.add_argument("--against-request", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.request is not None:
        payload = validate_request(_load_json(args.request))
    else:
        request = (
            _load_json(args.against_request)
            if args.against_request is not None
            else None
        )
        payload = validate_response(_load_json(args.response), request=request)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
