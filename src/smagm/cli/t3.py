"""Bounded synthetic T3 propagation smoke CLI (P0/P1 only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..anchors import AnchorBatch, AnchorGeometryBatch
from ..anchors.contracts import anchor_evidence_hash
from ..memory import PropagationConfig, initialize_seed_memory, propagate_memory
from ..state import PatientState, build_initial_patient_state, save_patient_state


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def synthetic_state() -> PatientState:
    geometry = AnchorGeometryBatch(
        ("anchor-0", "anchor-1"), torch.tensor([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), torch.eye(3).repeat(2, 1, 1),
        torch.tensor([[True, True, True], [True, True, True]]), torch.full((2, 3), 4.0),
        torch.ones(2, 1), torch.zeros(2, 1), (("context-a",), ("context-b",)),
        ((_digest("plane-a"),), (_digest("plane-b"),)), (_digest("anchor-a"), _digest("anchor-b")),
    )
    evidence = torch.tensor([[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, 0.5, 0.2]])
    appearance = torch.tensor([[0.2], [0.8]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    observability = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    digest = anchor_evidence_hash(patient_id="synthetic-patient", geometry=geometry, evidence=evidence, appearance=appearance, appearance_valid=valid, observability=observability)
    anchors = AnchorBatch("synthetic-patient", geometry, evidence, appearance, valid, observability, ("synthetic-mri",), digest)
    return build_initial_patient_state(
        patient_id="synthetic-patient", manifest_hash=_digest("t3-manifest"), config_hash=_digest("t3-config"),
        context_observation_ids=("context-a", "context-b"), cache_key_hashes=(_digest("cache-a"), _digest("cache-b")),
        anchors=anchors, field_config_hash=_digest("field-config"), field_model_hash=_digest("field-model"),
    )


def run(*, variant: str, rounds: int, output_dir: Path | None) -> dict[str, object]:
    state = synthetic_state()
    config = PropagationConfig(variant=variant, rounds=rounds, step_mm=0.75, maximum_structural_primitives=32, maximum_volumetric_primitives=32)
    memory, transactions = propagate_memory(
        state.memory, state.anchors, config=config,
        bounds_min_ras_mm=torch.tensor([-8.0, -8.0, -4.0]), bounds_max_ras_mm=torch.tensor([8.0, 8.0, 4.0]),
    )
    if memory.memory_hash != state.memory.memory_hash:
        from ..state import apply_memory_update
        state = apply_memory_update(state, memory)
    summary = {
        "schema": "smagm-t3-smoke-v1", "variant": variant, "rounds": rounds,
        "state_version": state.state_version, "memory_hash": state.memory.memory_hash,
        "primitive_count": state.memory.primitive_count,
        "accepted_per_round": [len(transaction.accepted_primitive_ids) for transaction in transactions],
        "transaction_hashes": [transaction.transaction_hash for transaction in transactions],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        save_patient_state(state, output_dir / "patient_state.pt")
        (output_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        with (output_dir / "propagation_transactions.jsonl").open("w", encoding="utf-8") as handle:
            for transaction in transactions:
                handle.write(json.dumps(transaction.__dict__, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded synthetic T3 propagation without T4 routing")
    parser.add_argument("--variant", choices=("p0", "p1"), default="p1")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(variant=args.variant, rounds=args.rounds, output_dir=args.output_dir)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items(): print(f"{key}: {value}")


if __name__ == "__main__":
    main()
