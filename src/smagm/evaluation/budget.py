"""Budget-quality summaries with exact declared observation counts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetCurve:
    budgets: tuple[int, ...]
    qualities: tuple[float, ...]
    trapezoid_auc: float


def summarize_budget_curve(budgets: tuple[int, ...], qualities: tuple[float, ...]) -> BudgetCurve:
    if len(budgets) != len(qualities) or len(budgets) < 2 or any(a >= b for a, b in zip(budgets, budgets[1:])):
        raise ValueError("budget curve requires aligned strictly increasing budgets")
    area = sum((b - a) * (qa + qb) * 0.5 for a, b, qa, qb in zip(budgets, budgets[1:], qualities, qualities[1:]))
    return BudgetCurve(budgets, qualities, float(area))
