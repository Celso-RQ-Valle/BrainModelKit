"""Shared cube option validation without dataframe dependencies."""

from collections.abc import Sequence
from itertools import combinations


def cube_options(columns, grupos, scores, target):
    """Normalize names and return all grouping subsets, overall first."""

    def names(value, label):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, Sequence) or not all(
            isinstance(name, str) for name in value
        ):
            raise TypeError(f"{label} must be a string or sequence of strings")
        if len(set(value)) != len(value):
            raise ValueError(f"{label} cannot contain duplicate columns")
        return list(value)

    groups = names(grupos, "grupos")
    score_names = names(scores, "scores")
    if not score_names:
        raise ValueError("scores must contain at least one column")
    if set(groups) & {"KS", "AUC", "Gini", "score", "auc", "gini"}:
        raise ValueError("Grouping columns conflict with metric output columns")
    if len(set(columns)) != len(columns):
        raise ValueError("Input cannot contain duplicate column names")
    missing = set([*groups, *score_names, target]).difference(columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    subsets = [
        list(subset)
        for size in range(len(groups) + 1)
        for subset in combinations(groups, size)
    ]
    return groups, score_names, subsets
