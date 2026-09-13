"""Pandas-native data quality filters."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._selection import SelectionResult, columns, groups, integer, number, resolve_frame


def numeric(
    df: pd.DataFrame, features: list[str], *, allow_missing: bool = False
) -> None:
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]) or pd.api.types.is_complex_dtype(
            df[f]
        ):
            raise ValueError(
                f"Feature {f!r} must be real numeric; encode categories explicitly"
            )
        values = df[f].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
            raise ValueError(
                f"Feature {f!r} must be finite; impute missing values first"
            )


def nonempty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame must contain at least one row")


def completeness(
    train_df: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    min_completeness: float = 0.7,
    group_by: str | Sequence[str] | None = None,
    require_all_groups: bool = False,
) -> SelectionResult:
    """Select by global non-missing fraction; group diagnostics are opt-in gates.

    See [Completeness](../../docs/feature_selection.md#completeness) for inputs,
    grouped diagnostics, and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    names = groups(train_df, group_by)
    number(min_completeness, "min_completeness", maximum=1)
    nonempty(train_df)
    if require_all_groups and not names:
        raise ValueError("require_all_groups requires group_by")
    rows = [
        {
            "feature": f,
            "completeness": float(train_df[f].notna().mean()),
            "missing_rate": float(train_df[f].isna().mean()),
            "selected": bool(train_df[f].notna().mean() >= min_completeness),
        }
        for f in features
    ]
    grouped = []
    if names:
        for key, frame in train_df.groupby(
            names, dropna=False, observed=True, sort=False
        ):
            keys = key if isinstance(key, tuple) else (key,)
            for row in rows:
                value = float(frame[row["feature"]].notna().mean())
                grouped.append(
                    {
                        **dict(zip(names, keys, strict=True)),
                        "feature": row["feature"],
                        "completeness": value,
                        "missing_rate": 1 - value,
                        "selected": value >= min_completeness,
                    }
                )
                if require_all_groups and value < min_completeness:
                    row["selected"] = False
    return SelectionResult(
        rows,
        "completeness",
        {
            "min_completeness": min_completeness,
            "require_all_groups": require_all_groups,
        },
        grouped_table=pd.DataFrame(grouped) if names else None,
    )


def variance_filter(
    train_df: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    min_variance: float = 0.0,
) -> SelectionResult:
    """Select population variance strictly above ``min_variance``.

    See [Variance](../../docs/feature_selection.md#variance) for assumptions and
    examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    number(min_variance, "min_variance")
    nonempty(train_df)
    numeric(train_df, features, allow_missing=True)
    rows = []
    for f in features:
        value = (
            float(train_df[f].var(ddof=0))
            if train_df[f].notna().any()
            else float("nan")
        )
        rows.append(
            {"feature": f, "variance": value, "selected": bool(value > min_variance)}
        )
    return SelectionResult(rows, "variance", {"min_variance": min_variance, "ddof": 0})


def cardinality(
    train_df: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
    min_unique: int = 2,
    max_unique: int | None = None,
    max_unique_ratio: float | None = None,
) -> SelectionResult:
    """Select using exact distinct counts and the all-row unique ratio.

    See [Cardinality](../../docs/feature_selection.md#cardinality) for options
    and examples.
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = columns(train_df, feature_cols)
    integer(min_unique, "min_unique", 0)
    if max_unique is not None:
        integer(max_unique, "max_unique", min_unique)
    if max_unique_ratio is not None:
        number(max_unique_ratio, "max_unique_ratio", maximum=1)
    nonempty(train_df)
    rows = []
    for f in features:
        count = int(train_df[f].nunique(dropna=True))
        ratio = count / len(train_df)
        rows.append(
            {
                "feature": f,
                "unique_count": count,
                "unique_ratio": ratio,
                "selected": count >= min_unique
                and (max_unique is None or count <= max_unique)
                and (max_unique_ratio is None or ratio <= max_unique_ratio),
            }
        )
    return SelectionResult(rows, "cardinality")
