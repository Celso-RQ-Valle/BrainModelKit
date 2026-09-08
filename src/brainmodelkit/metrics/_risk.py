"""Shared lightweight validation and output names for risk tables."""

RISK_COLUMNS = (
    "n_tile",
    "minimum_range",
    "maximum_range",
    "total_volume",
    "total_events",
    "total_non_events",
    "event_rate",
)


def validate_risk_options(
    columns: list[str],
    score_column: str,
    target_column: str,
    n_tiles: int,
    ascending: bool,
) -> None:
    """Validate configuration without computing any metrics."""
    if isinstance(n_tiles, bool) or not isinstance(n_tiles, int):
        raise TypeError("`n_tiles` must be an integer.")
    if n_tiles < 1:
        raise ValueError("`n_tiles` must be at least 1.")
    if not isinstance(ascending, bool):
        raise TypeError("`ascending` must be a boolean.")
    for name in (score_column, target_column):
        if not isinstance(name, str):
            raise TypeError("Score and target column names must be strings.")
        if name not in columns:
            raise KeyError(f"Missing required column: {name}")
        if columns.count(name) > 1:
            raise ValueError(f"Column name is ambiguous: {name}")
