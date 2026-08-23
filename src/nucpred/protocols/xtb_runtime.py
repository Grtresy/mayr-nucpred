"""Reusable fixed-geometry xTB execution, parsing, spin, and QC primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


HARTREE_TO_EV = 27.211386245988
QC_KEYS = (
    "frontier_orbital_min_hartree",
    "frontier_orbital_max_hartree",
    "cation_homo_min_hartree",
    "cation_homo_max_hartree",
    "vip_min_hartree",
    "vip_max_hartree",
    "vea_min_hartree",
    "vea_max_hartree",
    "ip_ea_gap_min_hartree",
    "max_abs_cm5_charge",
    "cm5_charge_sum_tolerance",
    "max_dipole_debye",
    "fukui_minus_sum_min",
    "fukui_minus_sum_max",
    "max_abs_fukui_atom",
    "max_abs_delta_e_solv_hartree",
    "max_abs_homo_shift_hartree",
)


@dataclass(frozen=True)
class PropertyResult:
    energy_hartree: float
    homo_ev: float
    lumo_ev: float
    dipole_debye: float
    cm5: tuple[float, ...]
    fukui_minus_raw: tuple[float, ...]


@dataclass(frozen=True)
class EnvironmentResult:
    property: PropertyResult
    vip_ev: float
    vea_ev: float
    cation_homo_ev: float


@dataclass(frozen=True)
class QualityControlResult:
    passed: bool
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return "ok" if self.passed else "; ".join(self.reasons)


@dataclass(frozen=True)
class Execution:
    stdout: str
    stderr: str
    returncode: int
    attempt: str
    wall_seconds: float
    command: tuple[str, ...]

    @property
    def output_sha256(self) -> str:
        return hashlib.sha256(
            (self.stdout + "\n" + self.stderr).encode("utf-8")
        ).hexdigest()


def _finite(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Expected a finite value, got {value!r}")
    return parsed


def _safe_extract_archive(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive) as handle:
        root = destination.resolve()
        members = handle.getmembers()
        for member in members:
            candidate = (destination / member.name).resolve()
            if root not in candidate.parents and candidate != root:
                raise ValueError(f"Unsafe xTB archive member: {member.name}")
        handle.extractall(destination, members=members, filter="data")
    binaries = sorted(destination.rglob("bin/xtb"))
    if len(binaries) != 1 or not binaries[0].is_file():
        raise FileNotFoundError("Could not identify exactly one xTB binary")
    return binaries[0].resolve()


def _xtb_environment(binary: Path) -> dict[str, str]:
    environment = os.environ.copy()
    distribution = binary.parents[1]
    environment.update(
        {
            "XTBHOME": distribution.as_posix(),
            "XTBPATH": (distribution / "share/xtb").as_posix(),
            "LD_LIBRARY_PATH": (distribution / "lib").as_posix(),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment


def _normalised_command(
    *,
    charge: int,
    uhf: int,
    solvent: str | None,
    mode: str,
    retry: bool,
) -> tuple[str, ...]:
    command = [
        "xtb",
        "molecule.xyz",
        "--gfn",
        "1",
        "--chrg",
        str(charge),
        "--uhf",
        str(uhf),
    ]
    if solvent is not None:
        command.extend(["--alpb", solvent])
    if mode:
        command.append(mode)
    command.extend(["--iterations", "1000" if retry else "500"])
    if retry:
        command.extend(["--etemp", "1000"])
    return tuple(command)


def _run_xtb(
    xyz_text: str,
    *,
    binary: Path,
    charge: int,
    uhf: int,
    solvent: str | None,
    mode: str,
    timeout_seconds: int,
) -> Execution:
    last: Execution | None = None
    for retry, attempt in ((False, "standard"), (True, "retry_etemp_1000")):
        with tempfile.TemporaryDirectory(
            prefix="nucpred_xtb_calc_"
        ) as raw_directory:
            workdir = Path(raw_directory)
            xyz_path = workdir / "molecule.xyz"
            xyz_path.write_text(xyz_text, encoding="utf-8")
            normalized = _normalised_command(
                charge=charge,
                uhf=uhf,
                solvent=solvent,
                mode=mode,
                retry=retry,
            )
            command = (binary.as_posix(), xyz_path.as_posix(), *normalized[2:])
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=workdir,
                    env=_xtb_environment(binary),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                stdout = completed.stdout
                stderr = completed.stderr
                returncode = int(completed.returncode)
            except subprocess.TimeoutExpired as error:
                stdout = str(error.stdout or "")
                stderr = str(error.stderr or "") + "\nTimeoutExpired"
                returncode = 124
            except OSError as error:
                stdout = ""
                stderr = f"{type(error).__name__}: {error}"
                returncode = 127
            last = Execution(
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                attempt=attempt,
                wall_seconds=time.perf_counter() - started,
                command=normalized,
            )
            combined = stdout + "\n" + stderr
            if (
                returncode == 0
                and "normal termination of xtb" in combined
                and "abnormal termination of xtb" not in combined
            ):
                return last
    if last is None:
        raise AssertionError("xTB execution loop did not run")
    raise RuntimeError(
        f"xTB failed after retry: rc={last.returncode}; "
        f"stderr={last.stderr.strip()[-500:]}"
    )


def _parse_property(
    output: str, atom_count: int, *, with_fukui: bool
) -> PropertyResult:
    energy = re.findall(r"TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh", output)
    homo = re.findall(r"([-+0-9.]+)\s+\(HOMO\)", output)
    lumo = re.findall(r"([-+0-9.]+)\s+\(LUMO\)", output)
    dipole = re.findall(r"total \(Debye\):\s+([-+0-9.Ee]+)", output)
    lines = output.splitlines()
    cm5: list[float] = []
    for line_index, line in enumerate(lines):
        if "Mulliken/CM5 charges" not in line:
            continue
        for charge_line in lines[line_index + 1 : line_index + 1 + atom_count]:
            parts = charge_line.split()
            if len(parts) < 3:
                break
            cm5.append(_finite(parts[2]))
        break
    fukui_minus: list[float] = []
    if with_fukui:
        for line_index, line in enumerate(lines):
            if "#        f(+)" not in line:
                continue
            for fukui_line in lines[
                line_index + 1 : line_index + 1 + atom_count
            ]:
                parts = fukui_line.split()
                if len(parts) < 4:
                    break
                fukui_minus.append(_finite(parts[2]))
            break
    if (
        not energy
        or not homo
        or not lumo
        or len(cm5) != atom_count
        or (with_fukui and len(fukui_minus) != atom_count)
    ):
        raise ValueError("Could not parse complete xTB property output")
    return PropertyResult(
        energy_hartree=_finite(energy[-1]),
        homo_ev=_finite(homo[-1]),
        lumo_ev=_finite(lumo[-1]),
        dipole_debye=_finite(dipole[-1]) if dipole else math.nan,
        cm5=tuple(cm5),
        fukui_minus_raw=tuple(fukui_minus),
    )


def _parse_ipea(output: str) -> tuple[float, float]:
    vip = re.findall(r"delta SCC IP \(eV\):\s+([-+0-9.Ee]+)", output)
    vea = re.findall(r"delta SCC EA \(eV\):\s+([-+0-9.Ee]+)", output)
    if not vip or not vea:
        raise ValueError("Could not parse xTB vertical IP/EA")
    return _finite(vip[-1]), _finite(vea[-1])


def _ledger_row(
    *,
    source_id: str,
    environment: str,
    calculation: str,
    charge: int,
    uhf: int,
    execution: Execution,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "environment": environment,
        "calculation": calculation,
        "charge": charge,
        "uhf": uhf,
        "attempt": execution.attempt,
        "returncode": execution.returncode,
        "wall_seconds": execution.wall_seconds,
        "normal_termination": True,
        "command": json.dumps(execution.command),
        "output_sha256": execution.output_sha256,
        "error": "",
    }


def _failure_ledger_row(
    *,
    source_id: str,
    environment: str,
    calculation: str,
    charge: int,
    uhf: int,
    solvent: str | None,
    mode: str,
    error: BaseException,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "environment": environment,
        "calculation": calculation,
        "charge": charge,
        "uhf": uhf,
        "attempt": "failed_after_retry",
        "returncode": -1,
        "wall_seconds": math.nan,
        "normal_termination": False,
        "command": json.dumps(
            _normalised_command(
                charge=charge,
                uhf=uhf,
                solvent=solvent,
                mode=mode,
                retry=True,
            )
        ),
        "output_sha256": "",
        "error": f"{type(error).__name__}: {error}",
    }


def _environment_calculations(
    xyz_text: str,
    *,
    source_id: str,
    binary: Path,
    charge: int,
    uhf: int,
    cation_uhf: int,
    solvent: str | None,
    timeout_seconds: int,
    atom_count: int,
) -> tuple[EnvironmentResult, list[dict[str, object]]]:
    label = solvent or "gas"
    executions: list[tuple[str, int, int, Execution]] = []
    failures: list[dict[str, object]] = []
    neutral = _run_xtb(
        xyz_text,
        binary=binary,
        charge=charge,
        uhf=uhf,
        solvent=solvent,
        mode="",
        timeout_seconds=timeout_seconds,
    )
    executions.append(("neutral_single_point", charge, uhf, neutral))
    properties = _parse_property(neutral.stdout, atom_count, with_fukui=False)
    try:
        fukui = _run_xtb(
            xyz_text,
            binary=binary,
            charge=charge,
            uhf=uhf,
            solvent=solvent,
            mode="--vfukui",
            timeout_seconds=timeout_seconds,
        )
        executions.append(("vfukui", charge, uhf, fukui))
        parsed_fukui = _parse_property(
            fukui.stdout, atom_count, with_fukui=True
        )
        properties = replace(
            properties, fukui_minus_raw=parsed_fukui.fukui_minus_raw
        )
    except (RuntimeError, ValueError) as error:
        properties = replace(
            properties, fukui_minus_raw=(math.nan,) * atom_count
        )
        failures.append(
            _failure_ledger_row(
                source_id=source_id,
                environment=label,
                calculation="vfukui",
                charge=charge,
                uhf=uhf,
                solvent=solvent,
                mode="--vfukui",
                error=error,
            )
        )
    vip_ev = math.nan
    vea_ev = math.nan
    try:
        ipea = _run_xtb(
            xyz_text,
            binary=binary,
            charge=charge,
            uhf=uhf,
            solvent=solvent,
            mode="--vipea",
            timeout_seconds=timeout_seconds,
        )
        executions.append(("vipea", charge, uhf, ipea))
        vip_ev, vea_ev = _parse_ipea(ipea.stdout)
    except (RuntimeError, ValueError) as error:
        failures.append(
            _failure_ledger_row(
                source_id=source_id,
                environment=label,
                calculation="vipea",
                charge=charge,
                uhf=uhf,
                solvent=solvent,
                mode="--vipea",
                error=error,
            )
        )
    cation_homo_ev = math.nan
    try:
        cation = _run_xtb(
            xyz_text,
            binary=binary,
            charge=charge + 1,
            uhf=cation_uhf,
            solvent=solvent,
            mode="",
            timeout_seconds=timeout_seconds,
        )
        executions.append(
            ("cation_single_point", charge + 1, cation_uhf, cation)
        )
        cation_properties = _parse_property(
            cation.stdout, atom_count, with_fukui=False
        )
        cation_homo_ev = cation_properties.homo_ev
    except (RuntimeError, ValueError) as error:
        failures.append(
            _failure_ledger_row(
                source_id=source_id,
                environment=label,
                calculation="cation_single_point",
                charge=charge + 1,
                uhf=cation_uhf,
                solvent=solvent,
                mode="",
                error=error,
            )
        )
    ledger = [
        _ledger_row(
            source_id=source_id,
            environment=label,
            calculation=calculation,
            charge=calculation_charge,
            uhf=calculation_uhf,
            execution=execution,
        )
        for calculation, calculation_charge, calculation_uhf, execution in executions
    ]
    ledger.extend(failures)
    return (
        EnvironmentResult(
            property=properties,
            vip_ev=vip_ev,
            vea_ev=vea_ev,
            cation_homo_ev=cation_homo_ev,
        ),
        ledger,
    )


def _minimal_uhf(atomic_numbers: Sequence[int], charge: int) -> int:
    electrons = sum(int(value) for value in atomic_numbers) - int(charge)
    if electrons <= 0:
        raise ValueError("Non-positive electron count")
    return electrons % 2


def _validated_neutral_uhf(
    atomic_numbers: Sequence[int], charge: int, smiles_radical_electrons: int
) -> tuple[int, str]:
    minimum = _minimal_uhf(atomic_numbers, charge)
    explicit = int(smiles_radical_electrons)
    if explicit < 0:
        raise ValueError("SMILES radical-electron count cannot be negative")
    if explicit % 2 != minimum:
        raise ValueError(
            "SMILES radical-electron count is inconsistent with geometry/charge "
            f"electron parity: radicals={explicit}, minimum={minimum}"
        )
    return explicit, "smiles_explicit_radicals" if explicit else "closed_shell"


def _quality_result(reasons: Sequence[str]) -> QualityControlResult:
    frozen = tuple(str(reason) for reason in reasons)
    return QualityControlResult(passed=not frozen, reasons=frozen)


def _range_issue(
    name: str, value: float, *, lower: float, upper: float
) -> str | None:
    if not math.isfinite(value):
        return f"{name} is non-finite"
    if value < lower or value > upper:
        return f"{name}={value:.8g} outside [{lower:.8g}, {upper:.8g}]"
    return None


def _property_qc(
    result: PropertyResult,
    *,
    formal_charge: int,
    thresholds: Mapping[str, object],
) -> QualityControlResult:
    reasons: list[str] = []
    for label, value in (
        ("HOMO_hartree", result.homo_ev / HARTREE_TO_EV),
        ("LUMO_hartree", result.lumo_ev / HARTREE_TO_EV),
    ):
        issue = _range_issue(
            label,
            value,
            lower=float(thresholds["frontier_orbital_min_hartree"]),
            upper=float(thresholds["frontier_orbital_max_hartree"]),
        )
        if issue:
            reasons.append(issue)
    if not math.isfinite(result.energy_hartree):
        reasons.append("total_energy_hartree is non-finite")
    dipole = float(result.dipole_debye)
    if (
        not math.isfinite(dipole)
        or dipole < 0
        or dipole > float(thresholds["max_dipole_debye"])
    ):
        reasons.append(
            f"dipole_debye={dipole:.8g} outside [0, "
            f"{float(thresholds['max_dipole_debye']):.8g}]"
        )
    charges = np.asarray(result.cm5, dtype=float)
    if not charges.size or not np.isfinite(charges).all():
        reasons.append("CM5 charge vector is empty or non-finite")
    else:
        maximum = float(np.abs(charges).max())
        if maximum > float(thresholds["max_abs_cm5_charge"]):
            reasons.append(
                f"max_abs_CM5={maximum:.8g} exceeds "
                f"{float(thresholds['max_abs_cm5_charge']):.8g}"
            )
        error = abs(float(charges.sum()) - float(formal_charge))
        if error > float(thresholds["cm5_charge_sum_tolerance"]):
            reasons.append(
                f"CM5_charge_sum_error={error:.8g} exceeds "
                f"{float(thresholds['cm5_charge_sum_tolerance']):.8g}"
            )
    return _quality_result(reasons)


def _fukui_qc(
    result: PropertyResult, *, thresholds: Mapping[str, object]
) -> QualityControlResult:
    values = np.asarray(result.fukui_minus_raw, dtype=float)
    if not values.size or not np.isfinite(values).all():
        return _quality_result(
            ["raw fukui-minus vector is empty or non-finite"]
        )
    reasons: list[str] = []
    issue = _range_issue(
        "raw_fukui_minus_sum",
        float(values.sum()),
        lower=float(thresholds["fukui_minus_sum_min"]),
        upper=float(thresholds["fukui_minus_sum_max"]),
    )
    if issue:
        reasons.append(issue)
    maximum = float(np.abs(values).max())
    if maximum > float(thresholds["max_abs_fukui_atom"]):
        reasons.append(
            f"max_abs_raw_fukui_minus={maximum:.8g} exceeds "
            f"{float(thresholds['max_abs_fukui_atom']):.8g}"
        )
    return _quality_result(reasons)


def _vipea_qc(
    result: EnvironmentResult, *, thresholds: Mapping[str, object]
) -> QualityControlResult:
    reasons: list[str] = []
    vip = result.vip_ev / HARTREE_TO_EV
    vea = result.vea_ev / HARTREE_TO_EV
    for label, value, lower_key, upper_key in (
        ("VIP_hartree", vip, "vip_min_hartree", "vip_max_hartree"),
        ("VEA_hartree", vea, "vea_min_hartree", "vea_max_hartree"),
    ):
        issue = _range_issue(
            label,
            value,
            lower=float(thresholds[lower_key]),
            upper=float(thresholds[upper_key]),
        )
        if issue:
            reasons.append(issue)
    hardness = vip - vea
    if math.isfinite(hardness) and hardness < float(
        thresholds["ip_ea_gap_min_hartree"]
    ):
        reasons.append(
            f"IP-EA_gap_hartree={hardness:.8g} below "
            f"{float(thresholds['ip_ea_gap_min_hartree']):.8g}"
        )
    return _quality_result(reasons)


def _cation_homo_qc(
    result: EnvironmentResult, *, thresholds: Mapping[str, object]
) -> QualityControlResult:
    issue = _range_issue(
        "cation_HOMO_hartree",
        result.cation_homo_ev / HARTREE_TO_EV,
        lower=float(thresholds["cation_homo_min_hartree"]),
        upper=float(thresholds["cation_homo_max_hartree"]),
    )
    return _quality_result([issue] if issue else [])


def _response_qc(
    environment: EnvironmentResult,
    *,
    gas: EnvironmentResult,
    environment_property_qc: QualityControlResult,
    gas_property_qc: QualityControlResult,
    thresholds: Mapping[str, object],
) -> QualityControlResult:
    reasons: list[str] = []
    if not gas_property_qc.passed:
        reasons.append(f"gas property QC failed: {gas_property_qc.reason}")
    if not environment_property_qc.passed:
        reasons.append(
            "environment property QC failed: "
            f"{environment_property_qc.reason}"
        )
    delta_e = (
        environment.property.energy_hartree - gas.property.energy_hartree
    )
    if (
        not math.isfinite(delta_e)
        or abs(delta_e)
        > float(thresholds["max_abs_delta_e_solv_hartree"])
    ):
        reasons.append(
            f"abs_delta_E_solv_hartree={abs(delta_e):.8g} exceeds "
            f"{float(thresholds['max_abs_delta_e_solv_hartree']):.8g}"
        )
    homo_shift = (
        environment.property.homo_ev - gas.property.homo_ev
    ) / HARTREE_TO_EV
    if (
        not math.isfinite(homo_shift)
        or abs(homo_shift)
        > float(thresholds["max_abs_homo_shift_hartree"])
    ):
        reasons.append(
            f"abs_HOMO_shift_hartree={abs(homo_shift):.8g} exceeds "
            f"{float(thresholds['max_abs_homo_shift_hartree']):.8g}"
        )
    return _quality_result(reasons)


def _xyz_text(
    atomic_numbers: Sequence[int], positions: Sequence[Sequence[float]]
) -> str:
    periodic_table = Chem.GetPeriodicTable()
    rows = [str(len(atomic_numbers)), "fixed-geometry xTB input"]
    for atomic_number, position in zip(atomic_numbers, positions, strict=True):
        rows.append(
            f"{periodic_table.GetElementSymbol(int(atomic_number)):<2s} "
            f"{float(position[0]): .10f} {float(position[1]): .10f} "
            f"{float(position[2]): .10f}"
        )
    return "\n".join(rows) + "\n"


def _tce_reference(
    binary: Path, *, smiles: str, timeout_seconds: int
) -> tuple[float, dict[str, object], str]:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 20260724
    if AllChem.EmbedMolecule(molecule, parameters) < 0:
        raise RuntimeError("Could not embed the TCE reference")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
    conformer = molecule.GetConformer()
    atomic_numbers = [
        atom.GetAtomicNum() for atom in molecule.GetAtoms()
    ]
    positions = [
        tuple(conformer.GetAtomPosition(index))
        for index in range(molecule.GetNumAtoms())
    ]
    xyz = _xyz_text(atomic_numbers, positions)
    execution = _run_xtb(
        xyz,
        binary=binary,
        charge=0,
        uhf=0,
        solvent=None,
        mode="",
        timeout_seconds=timeout_seconds,
    )
    parsed = _parse_property(
        execution.stdout, molecule.GetNumAtoms(), with_fukui=False
    )
    ledger = _ledger_row(
        source_id="reference:tetracyanoethylene",
        environment="gas",
        calculation="reference_single_point",
        charge=0,
        uhf=0,
        execution=execution,
    )
    return parsed.homo_ev / HARTREE_TO_EV, ledger, xyz
