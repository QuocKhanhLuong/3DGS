"""Deterministic patient-level paired summaries and bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class PairedSummary:
    patient_count: int
    mean_difference: float
    median_difference: float
    confidence_interval: tuple[float, float]
    bootstrap_seed: int


def paired_patient_summary(first: dict[str, float], second: dict[str, float], *, seed: int = 0, bootstrap_samples: int = 1000, confidence: float = 0.95) -> PairedSummary:
    if set(first) != set(second) or not first or bootstrap_samples <= 0 or not 0 < confidence < 1:
        raise ValueError("paired statistics require identical patients and valid bootstrap policy")
    patients = sorted(first); differences = [first[p] - second[p] for p in patients]
    if any(not math.isfinite(v) for v in differences):
        raise ValueError("paired differences must be finite")
    ordered = sorted(differences); middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    rng = random.Random(seed); means = []
    for _ in range(bootstrap_samples):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        means.append(sum(sample) / len(sample))
    means.sort(); alpha = (1 - confidence) / 2
    low = means[max(0, int(alpha * len(means)))]; high = means[min(len(means) - 1, int((1 - alpha) * len(means)) - 1)]
    return PairedSummary(len(patients), sum(differences) / len(differences), median, (low, high), seed)
