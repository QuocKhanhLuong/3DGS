"""Typed R4B headroom evidence and the fail-closed R5 admission gate.

The bounded NEXT-1 diagnostic is deliberately an evidence producer, not a
new policy or a second teacher.  This module only validates the sealed
identity/decision envelope that may be consumed by MAIN S2/S4; engineering
fixtures can record the same schema but can never mint a production permit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Sequence

from torch import Tensor


HEADROOM_SCHEMA = "pfgr-lite-headroom-decision-v1"
HEADROOM_STATUS = "INCONCLUSIVE"
MAIN_ACCEPTED = frozenset({"HEADROOM_CONFIRMED"})
# Scientific outcomes are deliberately closed to the four user-approved
# values.  Engineering capability and execution/error states are receipt
# status fields, never additional decision enums.
HEADROOM_DECISIONS = frozenset(
    {
        "HEADROOM_CONFIRMED",
        "CORRECTION_USEFUL_SELECTION_NOT_NEEDED",
        "NO_HEADROOM_OBSERVED",
        "INCONCLUSIVE",
    }
)
_SENTINELS = frozenset({"", "unknown", "unset", "none", "null", "missing", "stale"})
# The exact confirmation and the dense target-dependent decode are required
# to describe the same sealed action.  Keep this tolerance explicit in the
# retained evidence rather than allowing a caller to pick a favorable value
# while validating a permit.
_CONFIRMATION_GAIN_TOLERANCE = 1e-6


def _text(name: str, value: object, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if required and value.lower() in _SENTINELS:
        raise ValueError(f"{name} must be a complete identity")
    return value


def _finite(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class HeadroomDecision:
    """Immutable R4B decision bound to one frozen producer and cohort."""

    producer_compatibility_hash: str
    source_manifest_hash: str
    base_checkpoint_hash: str
    updater_checkpoint_hash: str
    split_hash: str
    subject_set_hash: str
    teacher_identity_hash: str
    candidate_pool_hash: str
    winner_action_id: str
    confirmation_action_id: str
    winner_confirmation_match: bool
    proposals_frozen_before_target: bool
    no_op_gain: float | None
    random_gain: float | None
    oracle_gain: float | None
    oracle_random_gain: float | None
    practical_margin: float
    uncertainty: float | None
    candidate_count: int
    query_count: int
    subject_count: int
    seeds: tuple[int, ...]
    delta_hashes: tuple[str, ...]
    saturation_definition: str
    decision: str
    scientific_status: str
    human_reviewed: bool
    reviewer: str | None
    created_at: str
    evidence_artifact_hash: str
    source_provenance_status: str
    precise: bool
    engineering_only: bool = False
    correction_only: bool = False
    # These paths/hashes are optional in early engineering receipts but are
    # mandatory for MAIN admission; they let the gate verify bytes rather than
    # trusting a user-editable ``decision``/``precise`` flag.
    evidence_artifact_path: str | None = None
    review_artifact_hash: str | None = None
    review_artifact_path: str | None = None
    options_hash: str | None = None
    schema_version: str = HEADROOM_SCHEMA
    frozen_model_digest: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != HEADROOM_SCHEMA:
            raise ValueError("unknown HeadroomDecision schema")
        for name in (
            "producer_compatibility_hash",
            "source_manifest_hash",
            "base_checkpoint_hash",
            "updater_checkpoint_hash",
            "split_hash",
            "subject_set_hash",
            "teacher_identity_hash",
            "candidate_pool_hash",
            "winner_action_id",
            "confirmation_action_id",
            "saturation_definition",
            "decision",
            "scientific_status",
            "created_at",
            "evidence_artifact_hash",
            "source_provenance_status",
        ):
            _text(name, getattr(self, name))
        if self.reviewer is not None:
            _text("reviewer", self.reviewer)
        if self.decision not in HEADROOM_DECISIONS:
            raise ValueError(f"unknown HeadroomDecision decision enum: {self.decision!r}")
        for name in (
            "winner_confirmation_match",
            "proposals_frozen_before_target",
            "human_reviewed",
            "precise",
            "engineering_only",
            "correction_only",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in (
            "no_op_gain",
            "random_gain",
            "oracle_gain",
            "oracle_random_gain",
            "practical_margin",
            "uncertainty",
        ):
            raw_value = getattr(self, name)
            if raw_value is None and name in {"no_op_gain", "random_gain", "oracle_gain", "oracle_random_gain", "uncertainty"}:
                continue
            value = _finite(name, raw_value)
            if name in {"practical_margin", "uncertainty"} and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("candidate_count", "query_count", "subject_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.candidate_count != 32:
            raise ValueError("NEXT-1 candidate_count is locked to 32")
        if self.query_count != 1024:
            raise ValueError("NEXT-1 query_count is locked to 1024")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("seeds must be a nonempty tuple")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must contain nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not isinstance(self.delta_hashes, tuple) or not self.delta_hashes:
            raise ValueError("delta_hashes must be a nonempty tuple")
        for value in self.delta_hashes:
            _text("delta_hash", value)
        if self.subject_count < 4:
            raise ValueError("headroom evidence requires at least four development subjects")
        if self.engineering_only and self.decision in MAIN_ACCEPTED:
            raise ValueError("engineering evidence cannot carry an accepted MAIN decision")
        if self.scientific_status.upper() == "INCONCLUSIVE" and self.decision in MAIN_ACCEPTED:
            raise ValueError("INCONCLUSIVE headroom evidence cannot be accepted")
        for name in ("evidence_artifact_path", "review_artifact_hash", "review_artifact_path", "options_hash", "frozen_model_digest"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")

    @property
    def digest(self) -> str:
        from .provenance import canonical_digest

        return canonical_digest(self._payload(), prefix="pfgr-lite-headroom-decision-v1|")

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                value = list(value)
            payload[item.name] = value
        return payload

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["digest"] = self.digest
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "HeadroomDecision":
        if not isinstance(values, Mapping):
            raise TypeError("headroom decision must be a mapping")
        allowed = {item.name for item in fields(cls)} | {"digest"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown HeadroomDecision keys: {sorted(unknown)}")
        payload = dict(values)
        payload.pop("digest", None)
        payload.setdefault("schema_version", HEADROOM_SCHEMA)
        payload["seeds"] = tuple(payload.get("seeds", ()))
        payload["delta_hashes"] = tuple(payload.get("delta_hashes", ()))
        return cls(**payload)


def validate_headroom_decision(
    value: HeadroomDecision | Mapping[str, Any],
    *,
    expected: Mapping[str, str] | None = None,
    allow_engineering: bool = False,
) -> HeadroomDecision:
    """Validate schema, evidence quality and exact producer identity joins."""

    declared_digest = value.get("digest") if isinstance(value, Mapping) else None
    decision = value if isinstance(value, HeadroomDecision) else HeadroomDecision.from_dict(value)
    if declared_digest is not None and declared_digest != decision.digest:
        raise ValueError("headroom decision digest does not match its immutable fields")
    if decision.engineering_only:
        if not allow_engineering:
            raise ValueError("engineering-only headroom evidence cannot authorize MAIN banking")
        return decision
    required_identity_keys = (
        "producer_compatibility_hash",
        "source_manifest_hash",
        "base_checkpoint_hash",
        "updater_checkpoint_hash",
        "split_hash",
        "subject_set_hash",
        "teacher_identity_hash",
        "frozen_model_digest",
    )
    if not isinstance(expected, Mapping):
        raise ValueError("MAIN headroom admission requires computed producer/checkpoint/source identities")
    missing_expected = [key for key in required_identity_keys if key not in expected]
    if missing_expected:
        raise ValueError(f"headroom expected identity set is incomplete: {missing_expected}")
    for key in required_identity_keys:
        expected_value = expected.get(key)
        if not isinstance(expected_value, str) or not expected_value.strip() or expected_value.lower() in _SENTINELS:
            raise ValueError(f"headroom expected identity {key} is unavailable")
        actual = getattr(decision, key, None)
        if actual != expected_value:
            raise ValueError(f"headroom identity mismatch for {key}: expected {expected_value!r}, got {actual!r}")
    expected_model_digest = expected.get("frozen_model_digest")
    if expected_model_digest is None:
        raise ValueError("headroom expected identity frozen_model_digest is unavailable")
    if not isinstance(expected_model_digest, str) or not expected_model_digest.strip() or decision.frozen_model_digest != expected_model_digest:
        raise ValueError("headroom frozen model identity does not match the live producer")
    if decision.source_provenance_status.upper() not in {"CLEAN", "VERIFIED", "PRODUCTION_VERIFIED"}:
        raise ValueError("headroom source provenance is missing, stale, dirty or unverified")
    if decision.decision not in MAIN_ACCEPTED:
        raise ValueError(f"headroom decision {decision.decision!r} does not authorize MAIN banking")
    if decision.scientific_status.upper() in {"INCONCLUSIVE", "PENDING", "STALE", "CORRECTION_ONLY"}:
        raise ValueError("headroom evidence is inconclusive or correction-only")
    if not decision.human_reviewed or decision.reviewer is None:
        raise ValueError("MAIN headroom admission requires explicit human review")
    if decision.evidence_artifact_path is None:
        raise ValueError("MAIN headroom admission requires a retained evidence artifact path")
    evidence_path = Path(decision.evidence_artifact_path)
    if not evidence_path.is_file():
        raise ValueError("headroom evidence artifact is missing or stale")
    actual_evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if actual_evidence_hash != decision.evidence_artifact_hash:
        raise ValueError("headroom evidence artifact hash does not match the decision")
    if decision.review_artifact_path is None or decision.review_artifact_hash is None:
        raise ValueError("MAIN headroom admission requires a retained human-review artifact")
    review_path = Path(decision.review_artifact_path)
    if not review_path.is_file() or hashlib.sha256(review_path.read_bytes()).hexdigest() != decision.review_artifact_hash:
        raise ValueError("headroom human-review artifact is missing or stale")
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("headroom human-review artifact is not valid JSON") from error
    if not isinstance(review, Mapping) or review.get("schema_version") != "pfgr-lite-headroom-review-v1":
        raise ValueError("headroom human-review artifact schema is unknown")
    review_required = {
        "schema_version",
        "decision",
        "reviewer",
        "evidence_artifact_hash",
        "options_hash",
        "identity_envelope",
    }
    if set(review) != review_required:
        raise ValueError("headroom human-review artifact must use the complete typed schema")
    if review.get("decision") not in {"APPROVED", "HEADROOM_CONFIRMED"}:
        raise ValueError("headroom human review does not approve MAIN admission")
    if review.get("reviewer") != decision.reviewer:
        raise ValueError("headroom reviewer identity does not match the decision")
    if review.get("evidence_artifact_hash") != decision.evidence_artifact_hash:
        raise ValueError("headroom review is not bound to the retained evidence bytes")
    if review.get("options_hash") != decision.options_hash:
        raise ValueError("headroom review is not bound to the frozen options")
    review_identity = review.get("identity_envelope")
    if not isinstance(review_identity, Mapping) or set(review_identity) != set(required_identity_keys):
        raise ValueError("headroom review identity envelope is incomplete")
    for key in required_identity_keys:
        if review_identity.get(key) != expected.get(key):
            raise ValueError(f"headroom review identity mismatch for {key}")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("headroom evidence artifact is not valid JSON") from error
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != "pfgr-lite-headroom-evidence-v1":
        raise ValueError("headroom evidence artifact schema is unknown")
    if evidence.get("privileged") is not True or evidence.get("target_dependent") is not True:
        raise ValueError("headroom evidence must be explicitly privileged and target-dependent")
    declared_confirmation_tolerance = evidence.get("confirmation_gain_tolerance")
    if (
        not isinstance(declared_confirmation_tolerance, (int, float))
        or not math.isfinite(float(declared_confirmation_tolerance))
        or float(declared_confirmation_tolerance) != _CONFIRMATION_GAIN_TOLERANCE
    ):
        raise ValueError("headroom evidence must declare the locked confirmation-gain tolerance")
    from .provenance import canonical_digest

    if decision.options_hash is None or evidence.get("options_hash") != decision.options_hash:
        raise ValueError("headroom evidence options hash does not match the immutable decision")
    evidence_options = evidence.get("options")
    if not isinstance(evidence_options, Mapping):
        raise ValueError("headroom evidence must retain the frozen evaluation options")
    if canonical_digest(dict(evidence_options), prefix="pfgr-lite-headroom-options-v1|") != decision.options_hash:
        raise ValueError("headroom options hash is not reproducible from retained options")
    declared_max_subjects = evidence_options.get("max_subjects")
    if not isinstance(declared_max_subjects, int) or isinstance(declared_max_subjects, bool) or declared_max_subjects < 4:
        raise ValueError("headroom options must declare at least four development subjects")
    if declared_max_subjects != decision.subject_count:
        raise ValueError("headroom options max_subjects does not match the retained cohort")
    if float(evidence_options.get("practical_margin", -1.0)) != float(decision.practical_margin):
        raise ValueError("headroom practical margin is not bound to the retained options")
    if int(evidence.get("subject_count", -1)) != decision.subject_count or int(evidence.get("candidate_count", -1)) != decision.candidate_count or int(evidence.get("query_count", -1)) != decision.query_count:
        raise ValueError("headroom evidence counts do not match the decision")
    subject_ids = evidence.get("subject_ids")
    if not isinstance(subject_ids, list) or len(subject_ids) != decision.subject_count:
        raise ValueError("headroom evidence subject identities are incomplete")
    if len(set(str(item) for item in subject_ids)) != len(subject_ids) or any(not str(item).strip() for item in subject_ids):
        raise ValueError("headroom evidence subject identities must be unique and nonempty")
    if canonical_digest(tuple(str(item) for item in subject_ids), prefix="pfgr-lite-headroom-subjects-v1|") != decision.subject_set_hash:
        raise ValueError("headroom evidence subject-set hash does not match the decision")
    if evidence.get("candidate_pool_hash") != decision.candidate_pool_hash:
        raise ValueError("headroom evidence candidate-pool hash does not match the decision")
    candidate_pool = evidence.get("candidate_pool")
    if not isinstance(candidate_pool, list) or canonical_digest(candidate_pool, prefix="pfgr-lite-headroom-candidate-pool-v1|") != decision.candidate_pool_hash:
        raise ValueError("headroom candidate pool bytes are not bound to the decision")
    if str(evidence.get("winner_action_id", "")) != decision.winner_action_id or str(evidence.get("confirmation_action_id", "")) != decision.confirmation_action_id:
        raise ValueError("headroom evidence winner/confirmation identity does not match the decision")
    if bool(evidence.get("winner_confirmation_match", False)) != decision.winner_confirmation_match:
        raise ValueError("headroom evidence confirmation status does not match the decision")
    subjects = evidence.get("subjects")
    if not isinstance(subjects, list) or len(subjects) != decision.subject_count:
        raise ValueError("headroom evidence lacks retained per-subject rows")
    dense_z0: list[float] = []
    dense_random: list[float] = []
    dense_oracle: list[float] = []
    dense_oracle_z0: list[float] = []
    dense_delta: list[float] = []
    for subject_index, subject in enumerate(subjects):
        if not isinstance(subject, Mapping):
            raise ValueError("headroom per-subject evidence row is malformed")
        if str(subject.get("subject_id", "")) != str(subject_ids[subject_index]):
            raise ValueError("headroom subject rows are not in the sealed cohort order")
        proposal_freeze = subject.get("proposal_freeze")
        if not isinstance(proposal_freeze, Mapping) or proposal_freeze.get("before_target") is not True:
            raise ValueError("headroom proposal set was not frozen before the target read")
        proposal_digest = proposal_freeze.get("proposal_digest")
        if not isinstance(proposal_digest, str) or not proposal_digest.strip() or proposal_digest.lower() in _SENTINELS:
            raise ValueError("headroom proposal freeze is missing its sealed proposal identity")
        screening = subject.get("screening", {})
        screen_rows = screening.get("rows") if isinstance(screening, Mapping) else None
        if not isinstance(screen_rows, list) or len(screen_rows) != decision.candidate_count:
            raise ValueError("headroom evidence screening rows are not dense")
        if any(_numeric_gain(row) is None for row in screen_rows if isinstance(row, Mapping)) or any(not isinstance(row, Mapping) for row in screen_rows):
            raise ValueError("headroom evidence contains an unmeasured candidate gain")
        subject_candidates = subject.get("candidate_pool")
        if not isinstance(subject_candidates, list) or len(subject_candidates) != decision.candidate_count:
            raise ValueError("headroom subject candidate pool is incomplete")
        candidate_ids: list[str] = []
        subject_pool_hash = subject.get("candidate_pool_hash")
        expected_subject_pool_hash = canonical_digest(
            subject_candidates,
            prefix="pfgr-lite-headroom-candidate-pool-v1|",
        )
        if subject_pool_hash != expected_subject_pool_hash:
            raise ValueError("headroom subject candidate-pool hash is not reproducible")
        for candidate_index, (candidate, row) in enumerate(zip(subject_candidates, screen_rows)):
            if not isinstance(candidate, Mapping):
                raise ValueError("headroom candidate identity row is malformed")
            if int(candidate.get("candidate_index", -1)) != candidate_index:
                raise ValueError("headroom candidate indices are not deterministic")
            candidate_id = str(candidate.get("action_id", ""))
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError("headroom candidate action identities must be unique")
            candidate_ids.append(candidate_id)
            if str(row.get("action_id", "")) != candidate_id:
                raise ValueError("headroom screening row is not bound to the frozen candidate")
            if not isinstance(candidate.get("point_ras_mm"), list) or len(candidate["point_ras_mm"]) != 3:
                raise ValueError("headroom candidate physical point coordinates are missing")
            if not isinstance(candidate.get("delta_hash"), str) or not candidate["delta_hash"]:
                raise ValueError("headroom candidate delta identity is missing")
            if row.get("point_ras_mm") != candidate.get("point_ras_mm"):
                raise ValueError("headroom screening point identity does not match the frozen candidate")
            if row.get("delta_hash") != candidate.get("delta_hash"):
                raise ValueError("headroom screening delta identity does not match the frozen candidate")
            candidate_action_digest = candidate.get("action_digest")
            if candidate_action_digest is not None and row.get("action_digest") != candidate_action_digest:
                raise ValueError("headroom screening action digest does not match the frozen candidate")
        # The exact confirmation is a retained measurement, not merely a
        # summary flag.  Require one row bound to the sealed winner's action,
        # physical point, delta, and action digest so an attacker cannot remove
        # the row and update the surrounding artifact hashes.
        confirmation = subject.get("confirmation")
        if not isinstance(confirmation, Mapping):
            raise ValueError("headroom exact confirmation record is missing")
        if confirmation.get("teacher_mode") != "exact_footprint":
            raise ValueError("headroom exact confirmation teacher mode is not retained")
        confirmation_rows = confirmation.get("rows")
        if not isinstance(confirmation_rows, list) or len(confirmation_rows) != 1:
            raise ValueError("headroom exact confirmation rows are incomplete")
        confirmation_row = confirmation_rows[0]
        if not isinstance(confirmation_row, Mapping):
            raise ValueError("headroom exact confirmation measurement row is malformed")
        oracle = subject.get("oracle")
        if not isinstance(oracle, Mapping):
            raise ValueError("headroom oracle row is missing")
        winner_id = str(oracle.get("winner_action_id", ""))
        confirmation_id = str(oracle.get("confirmation_action_id", ""))
        if not winner_id or winner_id != confirmation_id:
            raise ValueError("headroom subject winner/confirmation IDs differ")
        winner_index = next(
            (index for index, candidate in enumerate(subject_candidates)
             if isinstance(candidate, Mapping) and str(candidate.get("action_id", "")) == winner_id),
            None,
        )
        if winner_index is None:
            raise ValueError("headroom exact confirmation winner is outside the sealed candidate pool")
        winner_candidate = subject_candidates[winner_index]
        if str(confirmation_row.get("action_id", "")) != winner_id:
            raise ValueError("headroom exact confirmation action identity is not retained")
        if confirmation.get("winner_action_id") != winner_id or confirmation.get("confirmation_action_id") != confirmation_id or confirmation.get("same_winner") is not True:
            raise ValueError("headroom exact confirmation winner binding is incomplete")
        confirmation_candidate_index = confirmation_row.get("candidate_index")
        if (
            not isinstance(confirmation_candidate_index, int)
            or isinstance(confirmation_candidate_index, bool)
            or confirmation_candidate_index != winner_index
        ):
            raise ValueError("headroom exact confirmation candidate index is not bound to the winner")
        candidate_point = winner_candidate.get("point_ras_mm")
        measured_point = confirmation_row.get("point_ras_mm")
        if not isinstance(candidate_point, list) or measured_point != candidate_point:
            raise ValueError("headroom exact confirmation point identity does not match the sealed action")
        if confirmation_row.get("delta_hash") != winner_candidate.get("delta_hash"):
            raise ValueError("headroom exact confirmation delta identity does not match the sealed action")
        candidate_action_digest = winner_candidate.get("action_digest")
        if candidate_action_digest is not None and confirmation_row.get("action_digest") != candidate_action_digest:
            raise ValueError("headroom exact confirmation action digest does not match the sealed action")
        confirmation_gain = _numeric_gain(confirmation_row)
        if confirmation_gain is None or confirmation_row.get("finite_gain") is not True:
            raise ValueError("headroom exact confirmation measurement is missing or non-finite")
        footprint_voxels = confirmation_row.get("footprint_voxels")
        if not isinstance(footprint_voxels, int) or isinstance(footprint_voxels, bool) or footprint_voxels <= 0:
            raise ValueError("headroom exact confirmation footprint measurement is missing")
        measured_query_count = confirmation.get("actual_query_count")
        declared_query_count = confirmation.get("query_count")
        if measured_query_count != footprint_voxels or declared_query_count != footprint_voxels:
            raise ValueError("headroom exact confirmation query count is not bound to the retained footprint")
        if confirmation.get("configured_query_count") != decision.query_count:
            raise ValueError("headroom exact confirmation configured query count is not frozen")
        if confirmation_row.get("q_draws") not in (0, None) or confirmation.get("q_draws", 0) != 0:
            raise ValueError("headroom exact confirmation must declare zero sampled draws")
        controls = subject.get("random_controls")
        if not isinstance(controls, list) or len(controls) != len(decision.seeds):
            raise ValueError("headroom evidence random controls are incomplete")
        if any(not isinstance(item, Mapping) for item in controls):
            raise ValueError("headroom random control row is malformed")
        control_seeds = [item.get("seed") for item in controls]
        if tuple(control_seeds) != tuple(decision.seeds):
            raise ValueError("headroom random control seed identities do not match the frozen options")
        for control in controls:
            candidate_index = control.get("candidate_index")
            if (
                not isinstance(candidate_index, int)
                or isinstance(candidate_index, bool)
                or candidate_index < 0
                or candidate_index >= len(subject_candidates)
            ):
                raise ValueError("headroom random control candidate is outside the sealed candidate pool")
            if str(control.get("action_id", "")) != candidate_ids[candidate_index]:
                raise ValueError("headroom random control action is outside the sealed candidate pool")
            screening_gain = control.get("screening_gain")
            expected_screening_gain = _numeric_gain(screen_rows[candidate_index])
            if (
                not isinstance(screening_gain, (int, float))
                or not math.isfinite(float(screening_gain))
                or expected_screening_gain is None
                or abs(float(screening_gain) - expected_screening_gain) > 1e-12
            ):
                raise ValueError("headroom random control screening identity is not reproducible")
        control_gains = [item.get("gain") for item in controls]
        if len(control_gains) != len(decision.seeds) or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in control_gains):
            raise ValueError("headroom evidence random controls contain missing gains")
        if any(item.get("prediction_available") is not True for item in controls):
            raise ValueError("headroom random controls lack an actual target-free decode")
        oracle_value = oracle.get("gain") if isinstance(oracle, Mapping) else None
        if not isinstance(oracle_value, (int, float)) or not math.isfinite(float(oracle_value)):
            raise ValueError("headroom evidence oracle gain is missing")
        oracle_confirmation_gain = oracle.get("confirmation_gain")
        if (
            not isinstance(oracle_confirmation_gain, (int, float))
            or not math.isfinite(float(oracle_confirmation_gain))
            or abs(float(oracle_confirmation_gain) - float(confirmation_gain)) > 1e-12
        ):
            raise ValueError("headroom oracle confirmation gain is not bound to the retained measurement")
        if (
            abs(float(oracle_confirmation_gain) - float(oracle_value))
            > _CONFIRMATION_GAIN_TOLERANCE
        ):
            raise ValueError("headroom exact confirmation gain does not reconcile with the dense oracle effect")
        if not isinstance(oracle, Mapping) or oracle.get("prediction_available") is not True:
            raise ValueError("headroom oracle lacks an actual target-free decode")
        if oracle.get("same_winner") is not True:
            raise ValueError("headroom subject winner was not exactly confirmed")
        random_value = statistics.fmean(float(value) for value in control_gains)
        no_op = subject.get("no_op")
        no_op_value = no_op.get("gain") if isinstance(no_op, Mapping) else None
        if not isinstance(no_op_value, (int, float)) or not math.isfinite(float(no_op_value)):
            raise ValueError("headroom no-op gain is missing")
        delta_value = float(oracle_value) - random_value
        dense_z0.append(float(no_op_value))
        dense_random.append(random_value)
        dense_oracle.append(float(oracle_value))
        dense_oracle_z0.append(float(oracle_value) - float(no_op_value))
        dense_delta.append(delta_value)
        # The retained oracle action must be the deterministic winner of the
        # dense screening rows.  Recompute the same gain/point/index tie-break
        # used by NEXT-1 instead of trusting the summary winner flag.
        finite_screened = [
            (float(_numeric_gain(row)), index)
            for index, row in enumerate(screen_rows)
            if _numeric_gain(row) is not None
        ]
        if not finite_screened:
            raise ValueError("headroom screening has no finite candidate winner")
        def _screen_rank(item: tuple[float, int]) -> tuple[float, int, int]:
            gain, index = item
            point_id = subject_candidates[index].get("point_id")
            try:
                point_key = int(point_id) if point_id is not None else index
            except (TypeError, ValueError):
                point_key = index
            return gain, -point_key, -index

        expected_winner_index = max(finite_screened, key=_screen_rank)[1]
        if winner_id != candidate_ids[expected_winner_index]:
            raise ValueError("headroom oracle winner does not match dense screening ranking")
    if len(dense_delta) != decision.subject_count:
        raise ValueError("headroom evidence does not provide independent subject effects")
    if decision.no_op_gain is None or abs(float(decision.no_op_gain)) > 1e-12:
        raise ValueError("MAIN headroom admission requires an exact zero no-op gain")
    if any(abs(float(value)) > 1e-12 for value in dense_z0):
        raise ValueError("headroom retained no-op rows must have exactly zero gain")
    if abs(statistics.fmean(dense_z0) - float(decision.no_op_gain)) > 1e-12:
        raise ValueError("headroom no-op gain does not match retained per-subject rows")
    winner_ids = [str(subject.get("oracle", {}).get("winner_action_id", "")) for subject in subjects if isinstance(subject.get("oracle"), Mapping)]
    confirmation_ids = [str(subject.get("oracle", {}).get("confirmation_action_id", "")) for subject in subjects if isinstance(subject.get("oracle"), Mapping)]
    if len(winner_ids) != decision.subject_count or len(confirmation_ids) != decision.subject_count:
        raise ValueError("headroom evidence winner/confirmation rows are incomplete")
    if canonical_digest(tuple(winner_ids), prefix="pfgr-lite-headroom-winners-v1|") != decision.winner_action_id or canonical_digest(tuple(confirmation_ids), prefix="pfgr-lite-headroom-confirmations-v1|") != decision.confirmation_action_id:
        raise ValueError("headroom winner/confirmation IDs are not reproducible from retained rows")
    if any(winner != confirmation for winner, confirmation in zip(winner_ids, confirmation_ids)):
        raise ValueError("headroom winner and exact confirmation must be the same action per subject")
    expected_candidate_pool = [
        {"subject_id": str(subject.get("subject_id", "")), **dict(candidate)}
        for subject in subjects
        if isinstance(subject, Mapping) and isinstance(subject.get("candidate_pool"), list)
        for candidate in subject["candidate_pool"]
        if isinstance(candidate, Mapping)
    ]
    if candidate_pool != expected_candidate_pool:
        raise ValueError("headroom retained candidate pool does not match per-subject sealed pools")
    expected_delta_hashes: list[str] = []
    for subject in subjects:
        subject_candidates = subject.get("candidate_pool") if isinstance(subject, Mapping) else None
        oracle = subject.get("oracle") if isinstance(subject, Mapping) else None
        winner_id = str(oracle.get("winner_action_id", "")) if isinstance(oracle, Mapping) else ""
        winner_candidate = next(
            (candidate for candidate in subject_candidates
             if isinstance(candidate, Mapping) and str(candidate.get("action_id", "")) == winner_id),
            None,
        ) if isinstance(subject_candidates, list) else None
        if winner_candidate is None or not isinstance(winner_candidate.get("delta_hash"), str):
            raise ValueError("headroom winner delta identity is not retained")
        expected_delta_hashes.append(winner_candidate["delta_hash"])
    if tuple(decision.delta_hashes) != tuple(expected_delta_hashes):
        raise ValueError("headroom decision delta identities do not match retained winners")
    declared_metrics = evidence.get("dense_metrics")
    if not isinstance(declared_metrics, Mapping):
        raise ValueError("headroom dense metric arrays are missing")
    for key, actual in (("z0", dense_z0), ("random", dense_random), ("oracle", dense_oracle), ("oracle_minus_z0", dense_oracle_z0), ("oracle_minus_random", dense_delta)):
        declared = declared_metrics.get(key)
        if not isinstance(declared, list) or len(declared) != len(actual) or any(not isinstance(item, (int, float)) or not math.isfinite(float(item)) or abs(float(item) - expected_item) > 1e-12 for item, expected_item in zip(declared, actual)):
            raise ValueError(f"headroom evidence metric array {key!r} is not reproducible")
    if not bool(evidence.get("frozen_model_unchanged", False)):
        raise ValueError("headroom evidence shows a changed frozen model")
    if evidence.get("frozen_model_digest_before") != expected_model_digest or evidence.get("frozen_model_digest_after") != expected_model_digest:
        raise ValueError("headroom evidence frozen model digest does not match the live producer")
    if decision.random_gain is None or decision.oracle_gain is None or decision.oracle_random_gain is None:
        raise ValueError("headroom decision gains are missing")
    if abs(statistics.fmean(dense_random) - decision.random_gain) > 1e-12 or abs(statistics.fmean(dense_oracle) - decision.oracle_gain) > 1e-12 or abs(statistics.fmean(dense_delta) - decision.oracle_random_gain) > 1e-12:
        raise ValueError("headroom decision gains do not match retained per-subject evidence")
    declared_uncertainty = evidence.get("uncertainty")
    actual_uncertainty = statistics.stdev(dense_delta) / math.sqrt(len(dense_delta)) if len(dense_delta) > 1 else None
    if actual_uncertainty is None or decision.uncertainty is None or not isinstance(declared_uncertainty, (int, float)) or not math.isfinite(float(declared_uncertainty)) or abs(float(declared_uncertainty) - actual_uncertainty) > 1e-12 or abs(decision.uncertainty - actual_uncertainty) > 1e-12:
        raise ValueError("headroom uncertainty is not reproducible from independent subject rows")
    if not decision.precise or decision.subject_count < 32:
        raise ValueError("MAIN headroom admission requires adequately precise evidence from at least 32 subjects")
    if decision.correction_only:
        raise ValueError("correction-only evidence cannot authorize adaptive banking")
    if not decision.winner_confirmation_match or not decision.proposals_frozen_before_target:
        raise ValueError("headroom winner must be independently confirmed from proposals frozen before target reads")
    if decision.practical_margin <= 0.0:
        raise ValueError("final HEADROOM_CONFIRMED requires a positive predeclared practical margin")
    if decision.oracle_gain <= decision.practical_margin:
        raise ValueError("headroom must exceed Z0 by the frozen practical margin")
    if decision.oracle_random_gain <= decision.practical_margin:
        raise ValueError("headroom must exceed Random by the frozen practical margin")
    confidence = evidence.get("confidence_intervals")
    if not isinstance(confidence, Mapping) or confidence.get("method") != "normal_approximation_z_1.96" or float(confidence.get("level", 0.0)) != 0.95:
        raise ValueError("headroom confidence intervals are missing the locked method declaration")
    declared_lower_z0 = confidence.get("oracle_minus_z0_lower")
    declared_lower_random = confidence.get("oracle_minus_random_lower")
    expected_lower_z0 = _normal_lower_bound(dense_oracle_z0)
    expected_lower_random = _normal_lower_bound(dense_delta)
    if (
        not isinstance(declared_lower_z0, (int, float))
        or not isinstance(declared_lower_random, (int, float))
        or not math.isfinite(float(declared_lower_z0))
        or not math.isfinite(float(declared_lower_random))
        or abs(float(declared_lower_z0) - expected_lower_z0) > 1e-12
        or abs(float(declared_lower_random) - expected_lower_random) > 1e-12
        or expected_lower_z0 <= decision.practical_margin
        or expected_lower_random <= decision.practical_margin
    ):
        raise ValueError("headroom paired lower confidence bounds do not exceed the frozen practical margin")
    return decision


def require_main_headroom_decision(value: HeadroomDecision | Mapping[str, Any] | None, *, expected: Mapping[str, str] | None = None) -> HeadroomDecision:
    if value is None:
        raise ValueError("MAIN S2/S4 requires an accepted HeadroomDecision; R4A S1 success is not sufficient")
    return validate_headroom_decision(value, expected=expected, allow_engineering=False)


@dataclass(frozen=True)
class HeadroomOptions:
    """Bounded NEXT-1 settings; no architecture or loss knobs are exposed."""

    max_subjects: int = 4
    random_seeds: tuple[int, ...] = (17, 29, 41)
    candidate_count: int = 32
    query_count: int = 1024
    practical_margin: float = 0.0
    split_role: str = "validation"
    engineering_only: bool = False
    schema_version: str = "pfgr-lite-headroom-options-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "pfgr-lite-headroom-options-v1":
            raise ValueError("unknown HeadroomOptions schema")
        if self.max_subjects < 4:
            raise ValueError("headroom evaluation requires at least four development subjects")
        if tuple(self.random_seeds) != (17, 29, 41):
            raise ValueError("NEXT-1 random seeds are locked to 17,29,41")
        if self.candidate_count != 32 or self.query_count != 1024:
            raise ValueError("NEXT-1 requires candidate_count=32 and query_count=1024")
        if not math.isfinite(float(self.practical_margin)) or float(self.practical_margin) < 0.0:
            raise ValueError("practical_margin must be finite and nonnegative")
        if not isinstance(self.split_role, str) or not self.split_role.strip():
            raise ValueError("split_role must be nonempty")
        if not isinstance(self.engineering_only, bool):
            raise TypeError("engineering_only must be bool")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_subjects": self.max_subjects,
            "random_seeds": list(self.random_seeds),
            "candidate_count": self.candidate_count,
            "query_count": self.query_count,
            "practical_margin": self.practical_margin,
            "split_role": self.split_role,
            "engineering_only": self.engineering_only,
        }


def _metric_rows(path: Path | str | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return []
    try:
        import json

        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(loaded, Mapping):
        subjects = loaded.get("subjects")
        if isinstance(subjects, Mapping) and isinstance(subjects.get("rows"), list):
            return [dict(item) for item in subjects["rows"] if isinstance(item, Mapping)]
    return []


def _jsonl_rows(path: Path | str | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return []
    import json

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def _paired_gain(row: Mapping[str, Any]) -> float | None:
    improvement = row.get("improvement")
    if isinstance(improvement, Mapping):
        value = improvement.get("masked_charbonnier")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    value = row.get("oracle_route_gain", row.get("true_gain"))
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _route_attr(route: object, name: str, default: Any = None) -> Any:
    if isinstance(route, Mapping):
        return route.get(name, default)
    return getattr(route, name, default)


def _initial_state(route: object) -> object | None:
    state = _route_attr(route, "initial_state")
    if state is not None:
        return state
    trace = _route_attr(route, "completed_trace")
    states = tuple(_route_attr(trace, "states", ())) if trace is not None else tuple(_route_attr(route, "states", ()))
    return states[0] if states else _route_attr(route, "final_state")


def _clone_state(state: object) -> object:
    """Clone a typed state for independent controls without re-encoding Z0."""

    try:
        from .types import PFGRState, clone_dynamic_planes

        if isinstance(state, PFGRState):
            return PFGRState(
                planes=clone_dynamic_planes(state.planes),
                context_id=state.context_id,
                state_version=state.state_version,
                producer=state.producer,
                role=state.role,
            )
    except (ImportError, TypeError, ValueError):
        pass
    try:
        return copy.deepcopy(state)
    except Exception:
        # Callback engineering fixtures commonly use immutable strings or
        # mappings.  Returning the value is safe for those routes and keeps
        # the target-free freeze explicit rather than silently rebuilding Z0.
        return state


def _proposal_rows(proposal: object) -> list[object]:
    if proposal is None:
        return []
    if hasattr(proposal, "row") and hasattr(proposal, "point_ids"):
        return [proposal.row(0, index) for index in range(int(proposal.point_ids.shape[1]))]
    if isinstance(proposal, Mapping):
        values = proposal.get("candidates", proposal.get("proposals", ()))
        return list(values) if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else []
    if isinstance(proposal, Sequence) and not isinstance(proposal, (str, bytes)):
        return list(proposal)
    return []


def _action_id(action: object, index: int) -> str:
    value = getattr(action, "action_id", None)
    if isinstance(action, Mapping):
        value = action.get("action_id", action.get("id", value))
    return str(value) if value is not None else f"candidate-{index:03d}"


def _action_point_id(action: object, index: int) -> int | None:
    value = getattr(action, "point_id", None)
    if isinstance(action, Mapping):
        value = action.get("point_id", value)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _action_delta_hash(action: object) -> str:
    delta = getattr(action, "delta", None)
    if delta is None and isinstance(action, Mapping):
        delta = action.get("delta")
    if hasattr(delta, "detach"):
        from .provenance import tensor_digest

        return tensor_digest(delta.detach(), name="proposal_delta")
    declared = getattr(action, "delta_hash", None)
    if declared is None and isinstance(action, Mapping):
        declared = action.get("delta_hash")
    if isinstance(declared, str) and declared.strip():
        return declared
    # Callback fixtures may not expose a tensor.  Hashing the complete action
    # metadata remains an identity-only fallback, never a measured effect.
    from .provenance import canonical_digest

    return canonical_digest(_record_for_json(action), prefix="pfgr-lite-headroom-action-v1|")


def _record_for_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _record_for_json(item) for key, item in value.items() if str(key).lower() not in {"target", "prediction", "tensor"}}
    if isinstance(value, Tensor):
        from .provenance import tensor_digest

        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_digest(value.detach(), name="headroom")}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_record_for_json(item) for item in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _record_for_json(value.as_dict())
    return str(value)


def _point_coordinates(value: object) -> Any:
    """Serialize a candidate's physical point as numeric RAS-mm coordinates.

    Tensor digests are appropriate for large model/source values, but a point
    identity is part of the retained R4B candidate evidence and must remain
    inspectable/replayable without exposing target or observation bytes.
    """

    if isinstance(value, Tensor):
        if value.numel() != 3:
            return None
        values = value.detach().to(device="cpu", dtype=value.dtype).reshape(-1).tolist()
        return [float(item) for item in values]
    if isinstance(value, (tuple, list)) and len(value) == 3:
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _state_digest(state: object) -> str | None:
    value = getattr(state, "state_digest", None)
    if isinstance(value, str) and value:
        return value
    if isinstance(state, Mapping):
        value = state.get("state_digest")
        if isinstance(value, str) and value:
            return value
    return None


def _model_state_digest(model: object | None) -> str | None:
    if model is None or not hasattr(model, "state_dict"):
        return None
    # Reuse the package's canonical module-state algorithm everywhere.  A
    # second ad-hoc digest here would make the CLI's live-model identity
    # impossible to join to R4B evidence even for the same bytes.
    try:
        from torch import nn
        from .provenance import module_state_digest

        if isinstance(model, nn.Module):
            return module_state_digest(model)
    except Exception:
        pass
    return None


def _numeric_gain(row: Mapping[str, Any]) -> float | None:
    for key in ("true_gain", "raw_gain", "oracle_route_gain"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    improvement = row.get("improvement")
    if isinstance(improvement, Mapping):
        value = improvement.get("masked_charbonnier")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _normal_lower_bound(values: Sequence[float], *, confidence: float = 0.95) -> float:
    """Return a deterministic two-sided-normal lower confidence bound.

    NEXT-1 retains independent subject effects (not seed rows).  The normal
    approximation is deliberately named in the evidence schema so a future
    confirmatory analysis can replace it without silently changing this
    admission contract.
    """

    if not values:
        raise ValueError("cannot compute a confidence bound without subject effects")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("confidence-bound values must be finite")
    mean = statistics.fmean(float(value) for value in values)
    if len(values) == 1:
        standard_error = 0.0
    else:
        standard_error = statistics.stdev(float(value) for value in values) / math.sqrt(len(values))
    # Keep the confidence level explicit; only 95% is part of the locked
    # NEXT-1 artifact schema, and no user-facing post-hoc z-score is accepted.
    if confidence != 0.95:
        raise ValueError("NEXT-1 confidence level is locked to 0.95")
    return float(mean - 1.96 * standard_error)


def _actual_query_count(rows: Sequence[Mapping[str, Any]], *, configured: int, teacher_mode: str) -> int | None:
    """Extract measured teacher draws, distinguishing exact from sampled Q."""

    if teacher_mode == "exact_footprint":
        # Exact footprint has no sampled draw budget (q_draws remains zero on
        # the GainLabel), but it does perform one query for every unique
        # footprint voxel.  Report that measured footprint count explicitly so
        # a receipt cannot make an executed confirmation look like zero work.
        counts = [row.get("footprint_voxels") for row in rows if isinstance(row.get("footprint_voxels"), int)]
        if not counts:
            return None
        return int(sum(counts))
    values = [row.get("q_draws") for row in rows if isinstance(row.get("q_draws"), int)]
    return int(values[0]) if values and all(int(value) == int(values[0]) for value in values) else int(configured)


def _bind_measured_rows(
    rows: Sequence[object],
    candidates: Sequence[object],
    *,
    scope: str,
    seed: int,
    candidate_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Bind dense teacher rows to frozen candidates; missing gains stay missing."""

    if len(rows) != len(candidates):
        raise ValueError(f"{scope} measurement returned {len(rows)} rows for {len(candidates)} frozen candidates")
    if candidate_indices is not None:
        if len(candidate_indices) != len(candidates) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in candidate_indices
        ):
            raise ValueError(f"{scope} candidate_indices must align with nonnegative pool positions")
    bound: list[dict[str, Any]] = []
    for index, (candidate, value) in enumerate(zip(candidates, rows)):
        record = dict(value) if isinstance(value, Mapping) else _record_for_json(value)
        if not isinstance(record, dict):
            raise TypeError(f"{scope} measurement row must be a mapping")
        candidate_id = _action_id(candidate, index)
        declared = record.get("action_id")
        if declared is not None and str(declared) != candidate_id:
            raise ValueError(f"{scope} measurement action identity does not match the frozen candidate pool")
        record["action_id"] = candidate_id
        # Confirmation receives a singleton candidate list but must retain
        # the winner's original index in the sealed screening pool; otherwise
        # a reindexed row could be rebound to a different action after the
        # fact.  Screening naturally uses its local 0..N-1 positions.
        record["candidate_index"] = index if candidate_indices is None else int(candidate_indices[index])
        record["measurement_scope"] = scope
        record["measurement_seed"] = seed
        # Retain the immutable action/pool fields alongside the measured label;
        # a MAIN gate must be able to prove that an exact confirmation row is
        # for the sealed action/delta, not merely a hash-consistent summary.
        point = getattr(candidate, "point_ras_mm", None)
        if point is None and isinstance(candidate, Mapping):
            point = candidate.get("point_ras_mm", candidate.get("position_ras_mm"))
        record.setdefault("point_ras_mm", _point_coordinates(point))
        record.setdefault("delta_hash", _action_delta_hash(candidate))
        action_digest = getattr(candidate, "action_digest", None)
        if action_digest is None and isinstance(candidate, Mapping):
            action_digest = candidate.get("action_digest")
        if action_digest is not None:
            record.setdefault("action_digest", str(action_digest))
        gain = _numeric_gain(record)
        record["finite_gain"] = gain is not None
        bound.append(record)
    return bound


def _decode_applied(model: object | None, state: object, context: object | None, route: object, *, chunk_size: int = 1024) -> Tensor | None:
    if isinstance(state, Tensor):
        return state
    if isinstance(state, Mapping):
        for key in ("prediction", "final_prediction", "output"):
            value = state.get(key)
            if isinstance(value, Tensor):
                return value
    if model is not None and context is not None and hasattr(model, "decode_final") and hasattr(state, "planes"):
        return model.decode_final(state, context, chunk_size=chunk_size)
    for key in ("oracle_decode", "decode_state"):
        callback = _route_attr(route, key)
        if callable(callback):
            value = callback(state, context=context)
            if isinstance(value, Tensor):
                return value
    return None


def _write_norms(before_state: object, after_state: object, action: object | None = None) -> dict[str, Any]:
    from .provenance import tensor_digest

    correction = getattr(action, "delta", None)
    if correction is None and isinstance(action, Mapping):
        correction = action.get("delta")
    correction_record: dict[str, Any]
    if isinstance(correction, Tensor):
        detached = correction.detach()
        correction_record = {
            "available": True,
            "l2": float(detached.norm().item()),
            "max_abs": float(detached.abs().max().item()) if detached.numel() else 0.0,
            "numel": int(detached.numel()),
            "delta_hash": tensor_digest(detached, name="action_correction"),
        }
    else:
        correction_record = {"available": False, "reason": "action_delta_tensor_unavailable"}
    before_planes = getattr(before_state, "planes", None)
    after_planes = getattr(after_state, "planes", None)
    if before_planes is None or after_planes is None:
        return {
            "available": False,
            "reason": "typed_dynamic_planes_unavailable",
            "correction": correction_record,
        }

    rows: dict[str, Any] = {
        "available": True,
        "planes": {},
        "correction": correction_record,
        # Flat aliases make the measured action norm easy to audit alongside
        # the per-plane write norms without requiring consumers to infer a
        # tensor statistic from a nested payload.
        "correction_l2": correction_record.get("l2"),
        "correction_max_abs": correction_record.get("max_abs"),
    }
    for name in ("xy", "xz", "yz"):
        before = getattr(before_planes, name, None)
        after = getattr(after_planes, name, None)
        if not isinstance(before, Tensor) or not isinstance(after, Tensor) or before.shape != after.shape:
            rows["available"] = False
            rows["reason"] = "plane_shape_or_tensor_unavailable"
            return rows
        delta = after - before
        rows["planes"][name] = {
            "delta_l2": float(delta.norm().item()),
            "delta_max_abs": float(delta.abs().max().item()),
            "written_plane_l2": float(after.norm().item()),
            "written_plane_max_abs": float(after.abs().max().item()),
            "delta_hash": tensor_digest(delta.detach(), name=f"{name}_delta"),
        }
    rows["write_saturation"] = {"status": "not_applicable", "reason": "compact additive writer has no clipping or saturation operation"}
    return rows


def _run_headroom_evaluation_impl(inputs: Any, options: HeadroomOptions, output_dir: str | Path) -> Mapping[str, Any]:
    """Run one frozen-context, target-late NEXT-1 diagnostic.

    The four subjects are each encoded exactly once.  A single proposal batch
    is sealed from that initial state before the one deferred target join; the
    no-op, three random controls, fixed-Q screening, exact confirmation and
    stored-winner write all consume that same immutable context/bank.
    """

    if not isinstance(options, HeadroomOptions):
        raise TypeError("options must be HeadroomOptions")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"headroom output must be empty and exclusive: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    samples = tuple(getattr(inputs, "samples", ()))
    if len(samples) < options.max_subjects:
        raise ValueError("headroom evaluation requires the configured independent development cohort")

    from .experiments import (
        ExperimentOptions,
        _build_lattice,
        _context_for_sample,
        _load_policy,
        _prediction_for,
        _route_for_sample,
        _target_join,
        _target_parts,
    )
    from .metrics import paired_subject_metrics
    from .oracle import (
        OracleOptions,
        _measure_candidates,
        _oracle_advance,
        _oracle_continue_decision,
        _oracle_proposals,
    )
    from .provenance import canonical_digest, tensor_digest

    started = time.perf_counter()
    model = getattr(inputs, "model", None)
    model_digest_before = _model_state_digest(model)
    metadata = getattr(inputs, "metadata", {}) if isinstance(getattr(inputs, "metadata", {}), Mapping) else {}
    static_options = ExperimentOptions(
        scenario="noop",
        budget=0,
        max_subjects=1,
        seed=options.random_seeds[0],
        split_role=options.split_role,
        teacher_mode="exact_footprint",
        query_count=options.query_count,
        practical_margin=0.0,
        minimum_subjects=options.max_subjects,
        engineering_only=options.engineering_only,
    )
    screening_options = OracleOptions(
        mode="sampled_one",
        budget=1,
        candidate_count=options.candidate_count,
        teacher_mode="iid_fixed_q",
        query_count=options.query_count,
        max_subjects=1,
        seed=options.random_seeds[0],
        split_role=options.split_role,
        # OracleOptions requires an explicit independent-confirmation
        # declaration whenever screening uses iid_fixed_q.  The frozen
        # headroom loop performs that confirmation below itself, so this is a
        # declaration only and never triggers a nested target read.
        confirmation_mode="exact_footprint",
        confirmation_query_count=options.query_count,
        practical_margin=0.0,
        engineering_only=options.engineering_only,
    )
    confirmation_options = OracleOptions(
        mode="sampled_one",
        budget=1,
        candidate_count=1,
        teacher_mode="exact_footprint",
        query_count=options.query_count,
        max_subjects=1,
        seed=options.random_seeds[0] + 10_000,
        split_role=options.split_role,
        confirmation_mode="none",
        confirmation_query_count=options.query_count,
        practical_margin=0.0,
        engineering_only=options.engineering_only,
    )

    def _prediction_metric(initial: Tensor, final: Tensor | None, target_context: object, subject_id: str, scenario: str, budget: int) -> dict[str, Any] | None:
        if final is None:
            return None
        target, mask = _target_parts(target_context)
        return paired_subject_metrics(
            initial,
            final,
            target,
            mask,
            subject_id=subject_id,
            context_id=getattr(target_context, "context_id", None),
            scenario=scenario,
            budget=budget,
        )

    subject_records: list[dict[str, Any]] = []
    all_candidate_pool: list[dict[str, Any]] = []
    winner_ids: list[str] = []
    confirmation_ids: list[str] = []
    winner_delta_hashes: list[str] = []
    for subject_index, sample in enumerate(samples[: options.max_subjects]):
        subject_started = time.perf_counter()
        subject_id = str(getattr(sample, "subject_id", f"sample-{subject_index:04d}"))
        counters = metadata.get("data_counters", metadata.get("counters")) if isinstance(metadata, Mapping) else None
        context = _context_for_sample(inputs, sample)
        config = getattr(getattr(inputs, "execution", None), "config", getattr(inputs, "config", None))
        lattice = _build_lattice(inputs, context, model, config)
        policy = _load_policy(inputs, context, static_options, config)
        route, query, writer = _route_for_sample(inputs, sample, context, static_options, config, lattice, policy)
        initial_state = _initial_state(route)
        if initial_state is None:
            raise ValueError(f"headroom subject {subject_id} has no frozen initial PFGR state")
        try:
            initial_prediction = _prediction_for(model, route, context, final=False, options=static_options)
        except ValueError:
            if not options.engineering_only or not isinstance(_route_attr(route, "final_prediction"), Tensor):
                raise
            initial_prediction = _route_attr(route, "final_prediction")

        proposal = _oracle_proposals(
            inputs,
            context,
            initial_state,
            route,
            query,
            writer,
            lattice,
            screening_options,
            state_index=0,
            policy=policy,
        )
        all_rows = _proposal_rows(proposal)
        legal_rows: list[tuple[int, object]] = []
        for position, action in enumerate(all_rows):
            legal = getattr(action, "legal", True)
            if isinstance(action, Mapping):
                legal = action.get("legal", legal)
            if bool(legal):
                legal_rows.append((position, action))
        if len(legal_rows) < options.candidate_count:
            raise ValueError(f"headroom subject {subject_id} has only {len(legal_rows)} legal candidates; NEXT-1 requires 32")
        # The sampled pool is chosen once, deterministically, before any
        # target read.  Every control and the oracle use this exact pool.
        pool_rng = random.Random(options.random_seeds[0] + 1009 * subject_index)
        selected_positions = sorted(pool_rng.sample(range(len(legal_rows)), options.candidate_count)) if len(legal_rows) > options.candidate_count else list(range(options.candidate_count))
        candidates = [legal_rows[index][1] for index in selected_positions]
        candidate_records: list[dict[str, Any]] = []
        for pool_index, action in enumerate(candidates):
            point = getattr(action, "point_ras_mm", None)
            if point is None and isinstance(action, Mapping):
                point = action.get("point_ras_mm", action.get("position_ras_mm"))
            candidate_record = {
                "candidate_index": pool_index,
                "proposal_position": legal_rows[selected_positions[pool_index]][0],
                "action_id": _action_id(action, pool_index),
                "point_id": _action_point_id(action, pool_index),
                "point_ras_mm": _point_coordinates(point),
                "action_digest": getattr(action, "action_digest", None) if not isinstance(action, Mapping) else action.get("action_digest"),
                "delta_hash": _action_delta_hash(action),
                "legal": True,
            }
            candidate_records.append(candidate_record)
            all_candidate_pool.append({"subject_id": subject_id, **candidate_record})

        # Target-free state, route and proposal identities are sealed before
        # the deferred target callback.  Random choices are also frozen now.
        random_choices: list[dict[str, Any]] = []
        for seed in options.random_seeds:
            choice_rng = random.Random(seed + 1009 * subject_index)
            choice_index = choice_rng.randrange(options.candidate_count)
            random_choices.append({"seed": seed, "candidate_index": choice_index, "action_id": candidate_records[choice_index]["action_id"]})
        # Apply and decode every random control and the sealed winner before
        # the deferred target join.  The callback path intentionally receives
        # no target_context keyword; typed apply_scored_action likewise only
        # consumes the observation context, so target-derived values cannot
        # influence state mutation or decoding.
        def _apply(action: object, state: object) -> object | None:
            typed = hasattr(proposal, "row")
            decision = (
                _oracle_continue_decision(
                    proposal,
                    action,
                    policy,
                    step=int(getattr(state, "state_version", 0)),
                )
                if typed
                else None
            )
            return _oracle_advance(
                inputs,
                state,
                context,
                proposal,
                action,
                decision,
                writer=writer,
                state_index=0,
                route=route,
                target_context=None,
            )

        random_pending: list[dict[str, Any]] = []
        for choice in random_choices:
            index = int(choice["candidate_index"])
            action = candidates[index]
            applied = _apply(action, _clone_state(initial_state))
            prediction = (
                _decode_applied(
                    model,
                    applied,
                    context,
                    route,
                    chunk_size=getattr(config, "decode_chunk_size", 1024),
                )
                if applied is not None
                else None
            )
            random_pending.append(
                {
                    **choice,
                    "action_id": _action_id(action, index),
                    "state": applied,
                    "prediction": prediction,
                }
            )

        # Dense fixed-Q screening over the sealed 32-candidate pool happens
        # only after all target-free proposal/apply/decode work is complete.
        target_context = _target_join(
            inputs, sample, context, route, initial_prediction, static_options
        )
        target, target_mask = _target_parts(target_context)
        noop_metric = _prediction_metric(
            initial_prediction,
            initial_prediction,
            target_context,
            subject_id,
            "noop",
            0,
        )
        screened = _measure_candidates(
            inputs,
            route,
            candidates,
            target_context,
            context,
            lattice,
            screening_options,
            seed=screening_options.seed + subject_index,
            diagnostic_state=initial_state,
        )
        screening_rows = _bind_measured_rows(
            screened,
            candidates,
            scope="screening_iid_fixed_q",
            seed=screening_options.seed + subject_index,
        )
        finite_screened = [
            (gain, index, row)
            for index, row in enumerate(screening_rows)
            if (gain := _numeric_gain(row)) is not None
        ]
        winner_index = (
            max(
                finite_screened,
                key=lambda item: (
                    item[0],
                    -(
                        _action_point_id(candidates[item[1]], item[1])
                        if _action_point_id(candidates[item[1]], item[1]) is not None
                        else item[1]
                    ),
                    -item[1],
                ),
            )[1]
            if finite_screened
            else None
        )
        winner_action = candidates[winner_index] if winner_index is not None else None
        winner_action_id = (
            _action_id(winner_action, winner_index or 0)
            if winner_action is not None
            else "none"
        )
        winner_state = (
            _apply(winner_action, _clone_state(initial_state))
            if winner_action is not None
            else None
        )
        winner_prediction = (
            _decode_applied(
                model,
                winner_state,
                context,
                route,
                chunk_size=getattr(config, "decode_chunk_size", 1024),
            )
            if winner_state is not None
            else None
        )

        # The exact confirmation is measured against the same sealed winner;
        # it is intentionally run even when the sampled screen gain is
        # negative/near-zero so a selected identity can never disappear.
        confirmation_rows: list[dict[str, Any]] = []
        confirmation_gain: float | None = None
        confirmation_action_id = "none"
        if winner_action is not None:
            confirmed = _measure_candidates(
                inputs,
                route,
                [winner_action],
                target_context,
                context,
                lattice,
                confirmation_options,
                seed=confirmation_options.seed + subject_index,
                diagnostic_state=initial_state,
            )
            confirmation_rows = _bind_measured_rows(
                confirmed,
                [winner_action],
                scope="confirmation_exact_footprint",
                seed=confirmation_options.seed + subject_index,
                candidate_indices=[int(winner_index)],
            )
            if confirmation_rows:
                confirmation_action_id = str(confirmation_rows[0]["action_id"])
                confirmation_gain = _numeric_gain(confirmation_rows[0])
        match = winner_action is not None and confirmation_action_id == winner_action_id

        random_records: list[dict[str, Any]] = []
        for pending in random_pending:
            index = int(pending["candidate_index"])
            screen_row = screening_rows[index]
            metric = _prediction_metric(
                initial_prediction,
                pending["prediction"],
                target_context,
                subject_id,
                "random",
                1,
            )
            # Missing target-free apply/decode remains an explicit missing
            # dense result.  Never substitute sampled-screen or confirmation
            # rows for an actual final prediction metric.
            gain = _numeric_gain(metric) if metric is not None else None
            random_records.append(
                {
                    "seed": pending["seed"],
                    "candidate_index": index,
                    "action_id": pending["action_id"],
                    "screening_gain": _numeric_gain(screen_row),
                    "gain": gain,
                    "metric": metric,
                    "prediction_available": pending["prediction"] is not None,
                }
            )

        winner_metric = _prediction_metric(
            initial_prediction,
            winner_prediction,
            target_context,
            subject_id,
            "oracle",
            1,
        )
        random_gains = [record["gain"] for record in random_records]
        random_mean = (
            statistics.fmean(random_gains)
            if all(value is not None for value in random_gains)
            else None
        )
        oracle_gain = _numeric_gain(winner_metric) if winner_metric is not None else None
        z0_gain = _numeric_gain(noop_metric) if noop_metric is not None else None
        random_oracle_delta = (
            None
            if random_mean is None or oracle_gain is None
            else float(oracle_gain) - float(random_mean)
        )
        write_norms = (
            _write_norms(initial_state, winner_state, winner_action)
            if winner_state is not None
            else {"available": False, "reason": "winner_not_applied"}
        )
        subject_records.append(
            {
                "subject_id": subject_id,
                "context_id": getattr(context, "context_id", None),
                "mask_denominator": int(target_mask.sum().item()) if isinstance(target_mask, Tensor) else None,
                "producer_compatibility_hash": getattr(getattr(context, "producer", None), "compatibility_hash", getattr(getattr(context, "producer", None), "digest", None)),
                "z0_digest": tensor_digest(initial_prediction.detach(), name="headroom_z0"),
                "z0_state_digest": _state_digest(initial_state),
                "policy_hash": getattr(policy, "policy_hash", None),
                "candidate_pool": candidate_records,
                "candidate_pool_hash": canonical_digest(candidate_records, prefix="pfgr-lite-headroom-candidate-pool-v1|"),
                "screening": {"teacher_mode": "iid_fixed_q", "configured_query_count": options.query_count, "query_count": _actual_query_count(screening_rows, configured=options.query_count, teacher_mode="iid_fixed_q"), "actual_query_count": _actual_query_count(screening_rows, configured=options.query_count, teacher_mode="iid_fixed_q"), "mask_denominator": int(target_mask.sum().item()) if isinstance(target_mask, Tensor) else None, "rows": screening_rows},
                "confirmation": {"teacher_mode": "exact_footprint", "configured_query_count": options.query_count, "query_count": _actual_query_count(confirmation_rows, configured=options.query_count, teacher_mode="exact_footprint"), "actual_query_count": _actual_query_count(confirmation_rows, configured=options.query_count, teacher_mode="exact_footprint"), "rows": confirmation_rows, "winner_action_id": winner_action_id, "confirmation_action_id": confirmation_action_id, "same_winner": bool(match)},
                "random_controls": random_records,
                "no_op": {"gain": z0_gain, "metric": noop_metric},
                "oracle": {"screening_gain": None if winner_index is None else _numeric_gain(screening_rows[winner_index]), "confirmation_gain": confirmation_gain, "gain": oracle_gain, "metric": winner_metric, "query_count": options.query_count, "mask_denominator": int(target_mask.sum().item()) if isinstance(target_mask, Tensor) else None, "winner_action_id": winner_action_id, "confirmation_action_id": confirmation_action_id, "same_winner": bool(match), "prediction_available": winner_prediction is not None},
                "oracle_minus_random": random_oracle_delta,
                "write_norms": write_norms,
                "proposal_freeze": {"before_target": True, "proposal_digest": getattr(proposal, "proposal_digest", None)},
                "timing_seconds": {"subject_total": float(time.perf_counter() - subject_started)},
            }
        )
        winner_ids.append(winner_action_id)
        confirmation_ids.append(confirmation_action_id)
        if winner_action is not None:
            winner_delta_hashes.append(_action_delta_hash(winner_action))

    model_digest_after = _model_state_digest(model)
    no_op_values = [record["no_op"]["gain"] for record in subject_records if isinstance(record.get("no_op", {}).get("gain"), (int, float))]
    random_values = [record["random_controls"] for record in subject_records]
    random_means = [statistics.fmean([item["gain"] for item in controls]) if controls and all(item.get("gain") is not None for item in controls) else None for controls in random_values]
    oracle_values = [record["oracle"].get("gain") for record in subject_records]
    paired_delta = [record["oracle_minus_random"] for record in subject_records if record.get("oracle_minus_random") is not None]
    oracle_minus_z0 = [
        float(record["oracle"]["gain"]) - float(record["no_op"]["gain"])
        for record in subject_records
        if record["oracle"].get("gain") is not None and record["no_op"].get("gain") is not None
    ]
    uncertainty = statistics.stdev(paired_delta) / math.sqrt(len(paired_delta)) if len(paired_delta) > 1 else None
    subject_ids = tuple(record["subject_id"] for record in subject_records)
    candidate_pool_hash = canonical_digest(all_candidate_pool, prefix="pfgr-lite-headroom-candidate-pool-v1|")
    subject_set_hash = canonical_digest(subject_ids, prefix="pfgr-lite-headroom-subjects-v1|")
    producer_values = {str(record.get("producer_compatibility_hash")) for record in subject_records if record.get("producer_compatibility_hash")}
    producer_hash = next(iter(producer_values), None)
    if len(producer_values) > 1:
        raise ValueError("headroom subjects disagree on the frozen producer compatibility identity")
    def _metadata_identity(name: str, *, fallback: str | None = None) -> str:
        value = metadata.get(name) if isinstance(metadata, Mapping) else None
        if value is None and name == "split_hash":
            role_manifest = getattr(inputs, "role_manifest", None)
            value = getattr(role_manifest, "baseline_split_hash", None)
        if value is None:
            if not options.engineering_only:
                raise ValueError(f"production headroom requires computed {name} identity")
            value = fallback or f"engineering-{name}"
        return str(value)
    source_manifest_hash = _metadata_identity("source_manifest_hash", fallback="engineering-source-manifest")
    base_checkpoint_hash = _metadata_identity("base_checkpoint_hash", fallback="engineering-base")
    updater_checkpoint_hash = _metadata_identity("updater_checkpoint_hash", fallback="engineering-updater")
    split_hash = _metadata_identity("split_hash", fallback="engineering-split")
    teacher_identity_hash = canonical_digest({"screening": screening_options.as_dict(), "confirmation": confirmation_options.as_dict()}, prefix="pfgr-lite-headroom-teacher-v1|")
    source_status = str(metadata.get("source_provenance_status", "ENGINEERING_NONFINAL" if options.engineering_only else "UNKNOWN"))
    if producer_hash is None:
        producer_hash = _metadata_identity("producer_compatibility_hash", fallback="engineering-producer")
    options_hash = canonical_digest(options.as_dict(), prefix="pfgr-lite-headroom-options-v1|")
    dense_complete = all(
        len(record["screening"]["rows"]) == options.candidate_count
        and all(bool(row.get("finite_gain")) for row in record["screening"]["rows"])
        and len(record["confirmation"]["rows"]) == 1
        and record["oracle"].get("gain") is not None
        and all(item.get("gain") is not None for item in record["random_controls"])
        for record in subject_records
    )
    match_all = bool(subject_records) and all(record["oracle"].get("same_winner") for record in subject_records)
    evidence = {
        "schema_version": "pfgr-lite-headroom-evidence-v1",
        "privileged": True,
        "target_dependent": True,
        "options_hash": options_hash,
        "options": options.as_dict(),
        "subject_ids": list(subject_ids),
        "subject_count": len(subject_records),
        "candidate_count": options.candidate_count,
        "query_count": options.query_count,
        "candidate_pool_hash": candidate_pool_hash,
        "candidate_pool": all_candidate_pool,
        "subjects": subject_records,
        "confirmation_gain_tolerance": _CONFIRMATION_GAIN_TOLERANCE,
        "winner_action_id": canonical_digest(tuple(winner_ids), prefix="pfgr-lite-headroom-winners-v1|"),
        "confirmation_action_id": canonical_digest(tuple(confirmation_ids), prefix="pfgr-lite-headroom-confirmations-v1|"),
        "winner_confirmation_match": match_all,
        "proposals_frozen_before_target": all(record["proposal_freeze"].get("before_target") for record in subject_records),
        "dense_metrics": {"z0": no_op_values, "random": random_means, "oracle": oracle_values, "oracle_minus_z0": oracle_minus_z0, "oracle_minus_random": paired_delta},
        "independent_subject_count": len(paired_delta),
        "uncertainty": uncertainty,
        "confidence_intervals": {
            "method": "normal_approximation_z_1.96",
            "level": 0.95,
            "oracle_minus_z0_lower": _normal_lower_bound(oracle_minus_z0) if len(oracle_minus_z0) == len(subject_records) and oracle_minus_z0 else None,
            "oracle_minus_random_lower": _normal_lower_bound(paired_delta) if len(paired_delta) == len(subject_records) and paired_delta else None,
        },
        "precision": {"finite_dense_rows": dense_complete, "independent_subjects": len(paired_delta), "minimum_for_main": 32},
        "frozen_model_digest_before": model_digest_before,
        "frozen_model_digest_after": model_digest_after,
        "frozen_model_unchanged": model_digest_before == model_digest_after,
        "write_saturation": {"status": "not_applicable", "reason": "compact additive writer has no clipping or saturation operation"},
        "saturation_definition": "not_applicable:additive_compact_writer_no_clipping_v1",
        "timing_seconds": {"end_to_end": float(time.perf_counter() - started)},
        "scientific_status": "INCONCLUSIVE",
    }
    evidence_path = destination / "next1_evidence.json"
    evidence_path.write_text(json.dumps(_record_for_json(evidence), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    decision_payload = HeadroomDecision(
        producer_compatibility_hash=producer_hash,
        source_manifest_hash=source_manifest_hash,
        base_checkpoint_hash=base_checkpoint_hash,
        updater_checkpoint_hash=updater_checkpoint_hash,
        split_hash=split_hash,
        subject_set_hash=subject_set_hash,
        teacher_identity_hash=teacher_identity_hash,
        candidate_pool_hash=candidate_pool_hash,
        winner_action_id=evidence["winner_action_id"],
        confirmation_action_id=evidence["confirmation_action_id"],
        winner_confirmation_match=match_all,
        proposals_frozen_before_target=bool(evidence["proposals_frozen_before_target"]),
        no_op_gain=statistics.fmean(no_op_values) if no_op_values and len(no_op_values) == len(subject_records) else None,
        random_gain=statistics.fmean([value for value in random_means if value is not None]) if random_means and all(value is not None for value in random_means) else None,
        oracle_gain=statistics.fmean([float(value) for value in oracle_values if value is not None]) if oracle_values and all(value is not None for value in oracle_values) else None,
        oracle_random_gain=statistics.fmean(paired_delta) if paired_delta and len(paired_delta) == len(subject_records) else None,
        practical_margin=options.practical_margin,
        uncertainty=float(uncertainty) if uncertainty is not None else None,
        candidate_count=options.candidate_count,
        query_count=options.query_count,
        subject_count=len(subject_records),
        seeds=options.random_seeds,
        delta_hashes=tuple(winner_delta_hashes or ["engineering-no-winner-delta"]),
        saturation_definition="not_applicable:additive_compact_writer_no_clipping_v1",
        decision="INCONCLUSIVE",
        scientific_status="INCONCLUSIVE",
        human_reviewed=False,
        reviewer=None,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        evidence_artifact_hash=evidence_hash,
        source_provenance_status=source_status,
        precise=False,
        engineering_only=options.engineering_only,
        correction_only=False,
        evidence_artifact_path=str(evidence_path),
        options_hash=options_hash,
        frozen_model_digest=model_digest_before,
    )
    confirmation_query_counts = [
        int(record["confirmation"]["actual_query_count"])
        for record in subject_records
        if isinstance(record.get("confirmation", {}).get("actual_query_count"), int)
    ]
    confirmation_query_total = sum(confirmation_query_counts)
    summary = {
        "schema_version": "pfgr-lite-headroom-result-v2",
        "status": "SOFTWARE_PASS",
        "scientific_status": "INCONCLUSIVE",
        "privileged": True,
        "target_dependent": True,
        "options": options.as_dict(),
        "subject_ids": list(subject_ids),
        "candidate_pool_hash": candidate_pool_hash,
        "subjects": subject_records,
        "dense_metrics": evidence["dense_metrics"],
        "screening": {"candidate_count": options.candidate_count, "query_count": options.query_count, "teacher_mode": "iid_fixed_q", "rows": sum(len(record["screening"]["rows"]) for record in subject_records)},
        "confirmation": {"mode": "exact_footprint", "configured_query_count": options.query_count, "q_draws": 0, "query_count": confirmation_query_total, "actual_query_count": confirmation_query_total, "per_subject_query_count": confirmation_query_counts, "same_winner": match_all, "winner_action_id": evidence["winner_action_id"], "confirmation_action_id": evidence["confirmation_action_id"]},
        "correction_write_norms": [record["write_norms"] for record in subject_records],
        "write_saturation": evidence["write_saturation"],
        "saturation_definition": evidence["saturation_definition"],
        "uncertainty": {"paired_standard_error": uncertainty, "independent_subject_count": len(paired_delta)},
        "timing_seconds": evidence["timing_seconds"],
        "frozen_model": {"before": model_digest_before, "after": model_digest_after, "unchanged": model_digest_before == model_digest_after},
        "artifacts": {"evidence": str(evidence_path), "decision": str(destination / "headroom_decision.json")},
        "headroom_decision": decision_payload.as_dict(),
        "scientific_note": "NEXT-1 is an early four-subject diagnostic and remains INCONCLUSIVE; it cannot authorize MAIN R5.",
    }
    (destination / "headroom_metrics.json").write_text(json.dumps(_record_for_json(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (destination / "headroom_decision.json").write_text(json.dumps(decision_payload.as_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "software_status": "SOFTWARE_PASS",
        "scientific_status": "INCONCLUSIVE",
        "privileged": True,
        "target_dependent": True,
        "subject_count": len(subject_records),
        "metrics_path": destination / "headroom_metrics.json",
        "evidence_path": evidence_path,
        "decision_path": destination / "headroom_decision.json",
        "headroom_decision": decision_payload.as_dict(),
    }


def run_headroom_evaluation(inputs: Any, options: HeadroomOptions, output_dir: str | Path) -> Mapping[str, Any]:
    """Execute NEXT-1 with model evaluation/no-grad semantics restored on exit."""

    from .experiments import _service_execution

    # A historical width-128 checkpoint is marked on the live model during
    # hydration.  Its serialized PFGR config intentionally remains unchanged,
    # so guard the direct service boundary against an accidental production
    # invocation that supplies ``engineering_only=False``.
    if bool(getattr(getattr(inputs, "model", None), "_engineering_only", False)) and not options.engineering_only:
        raise ValueError("engineering-hydrated model is restricted to explicit engineering-only headroom evaluation")

    with _service_execution(getattr(inputs, "model", None)):
        return _run_headroom_evaluation_impl(inputs, options, output_dir)


__all__ = [
    "HEADROOM_SCHEMA",
    "HeadroomDecision",
    "HeadroomOptions",
    "require_main_headroom_decision",
    "validate_headroom_decision",
    "run_headroom_evaluation",
]
