"""Spark-native validation of external-provider returns."""

from __future__ import annotations

import re
from functools import reduce
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


BANDS = ["Missing", *[f"D{i}" for i in range(1, 11)]]


def validate_external_dataframe(
    df_sent: DataFrame,
    df_received: DataFrame,
    key_cols: list[str],
    date_col: str,
    score_col: str,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Validate provider coverage and score stability without collecting raw rows.

    Keys with any SQL null component are excluded from coverage denominators.
    Duplicate rows remain in every received-score analysis. Numeric scores are
    required; SQL null/NaN are missing, infinities are invalid. Dates support
    Spark date/timestamp types and strict YYYY-MM-DD strings (session timezone).
    Baseline defaults to the earliest valid month, even without valid scores.
    Percentiles use Spark percentile_approx with accuracy 10000 (relative rank
    error 1/10000). Bands are (-inf, p10], (p10, p20], ..., (p90, inf).
    Ties stay together; empty bands are preserved. Histogram density integrates
    to one over finite scores in [0, 1000], using 100-point bins.

    Return a dictionary of matching, scores, dates, monthly_volume, baseline,
    cutoffs, band_definitions, monthly_bands, histogram, figures, warnings and
    stability_status. Tables are lists of dictionaries collected after Spark
    aggregation only. Percentages use 0..100; zero denominators return None.
    Matplotlib and IPython are optional: figures are returned when installed,
    and sections are displayed in an active notebook. See docs/external_validation.md.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DateType,
        NumericType,
        StringType,
        TimestampNTZType,
        TimestampType,
    )

    def col(name):
        return F.col("`" + name.replace("`", "``") + "`")

    if (
        not isinstance(key_cols, list)
        or not key_cols
        or any(not isinstance(k, str) or not k for k in key_cols)
    ):
        raise ValueError("key_cols must be a nonempty list of column names")
    if len(set(key_cols)) != len(key_cols):
        raise ValueError("key_cols must contain unique column names")
    if any(not isinstance(name, str) or not name for name in (date_col, score_col)):
        raise ValueError("date_col and score_col must be nonempty column names")
    for frame, required in (
        (df_sent, key_cols),
        (df_received, [*key_cols, date_col, score_col]),
    ):
        if len(frame.columns) != len(set(frame.columns)):
            raise ValueError("DataFrames must have unique column names")
        if missing := set(required) - set(frame.columns):
            raise ValueError(f"Missing required columns: {sorted(missing)}")
    for key in key_cols:
        if df_sent.schema[key].dataType != df_received.schema[key].dataType:
            raise ValueError(
                f"Key column {key!r} must have the same type in both datasets"
            )
    if not isinstance(df_received.schema[score_col].dataType, NumericType):
        raise ValueError("score_col must be numeric; malformed strings are not coerced")
    dtype = df_received.schema[date_col].dataType
    if not isinstance(dtype, (DateType, TimestampType, TimestampNTZType, StringType)):
        raise ValueError("date_col must be a date, timestamp, or YYYY-MM-DD string")
    if baseline is not None and (
        not isinstance(baseline, str)
        or not re.fullmatch(r"[0-9]{4}(0[1-9]|1[0-2])", baseline)
        or baseline.startswith("0000")
    ):
        raise ValueError("baseline must be a valid YYYYMM string or None")

    def pct(n, total):
        return 100.0 * n / total if total else None

    def key_summary(frame):
        aliases = [f"key_{i}" for i in range(len(key_cols))]
        frame = frame.select(
            *[col(k).alias(a) for k, a in zip(key_cols, aliases, strict=True)]
        )
        valid = reduce(lambda a, b: a & b, [col(k).isNotNull() for k in aliases])
        groups = frame.groupBy(*aliases).count()
        totals = (
            groups.agg(
                F.sum("count").alias("rows"),
                F.count("*").alias("distinct_keys"),
                F.sum(F.when(~valid, col("count")).otherwise(0)).alias("null_key_rows"),
                F.sum(F.when(col("count") > 1, 1).otherwise(0)).alias(
                    "duplicate_key_groups"
                ),
                F.sum(col("count") - 1).alias("duplicate_key_rows"),
                F.sum(F.when(valid, 1).otherwise(0)).alias("valid_distinct_keys"),
            )
            .first()
            .asDict()
        )
        return {k: int(v or 0) for k, v in totals.items()}, groups.filter(valid).select(
            *aliases
        )

    sent, sent_keys = key_summary(df_sent)
    received, received_keys = key_summary(df_received)
    matched = sent_keys.join(received_keys, sent_keys.columns, "inner").count()
    matching = dict(
        submitted=sent,
        received=received,
        matched_distinct_keys=matched,
        missing_submitted_keys=sent["valid_distinct_keys"] - matched,
        unexpected_received_keys=received["valid_distinct_keys"] - matched,
        submitted_match_percentage=pct(matched, sent["valid_distinct_keys"]),
        received_match_percentage=pct(matched, received["valid_distinct_keys"]),
    )
    date = col(date_col)
    if isinstance(dtype, StringType):
        date = F.when(
            date.rlike(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"),
            F.try_to_timestamp(date, F.lit("yyyy-MM-dd")),
        )
    data = df_received.select(
        col(score_col).alias("score"),
        F.date_format(date, "yyyyMM").alias("cohort"),
        col(date_col).isNull().alias("date_missing"),
    )
    score = col("score")
    missing = score.isNull() | F.isnan(score)
    invalid = ~missing & (F.abs(score) == float("inf"))
    valid = ~missing & ~invalid

    def count_if(condition, name):
        return F.count(F.when(condition, 1)).alias(name)

    counts = (
        data.agg(
            count_if(missing, "missing"),
            count_if(invalid, "invalid"),
            count_if(valid, "valid"),
            count_if(valid & (score < 0), "below_range"),
            count_if(valid & (score > 1000), "above_range"),
            count_if(col("date_missing"), "missing_dates"),
            count_if(col("cohort").isNull() & ~col("date_missing"), "invalid_dates"),
        )
        .first()
        .asDict()
    )
    dates = {k: counts.pop(k) for k in ("missing_dates", "invalid_dates")}
    counts["missing_percentage"] = pct(counts["missing"], received["rows"])
    monthly = [
        r.asDict()
        for r in data.filter(col("cohort").isNotNull())
        .groupBy("cohort")
        .count()
        .orderBy("cohort")
        .collect()
    ]
    selected = (
        baseline
        if baseline is not None
        else (monthly[0]["cohort"] if monthly else None)
    )
    warnings = []
    cutoffs = []
    status = "available"
    if selected is None:
        status = "No valid monthly cohorts are available."
    elif selected not in {r["cohort"] for r in monthly}:
        status = f"Baseline {selected} does not exist in received cohorts."
    else:
        cutoffs = (
            data.filter((col("cohort") == selected) & valid)
            .agg(
                F.percentile_approx(
                    "score", [i / 10 for i in range(1, 10)], 10000
                ).alias("cutoffs")
            )
            .first()["cutoffs"]
            or []
        )
        if not cutoffs:
            status = (
                f"Baseline {selected} has no valid scores; "
                "deciles cannot be calculated."
            )
        elif data.filter(col("cohort").isNotNull() & invalid).limit(1).count():
            status = (
                "Invalid scores in dated rows prevent complete eleven-band assignment."
            )
    definitions = []
    if cutoffs:
        definitions = [
            dict(band="Missing", lower=None, upper=None, interval="SQL null or NaN")
        ]
        edges = [float("-inf"), *cutoffs, float("inf")]
        definitions += [
            dict(
                band=f"D{i + 1}",
                lower=edges[i],
                upper=edges[i + 1],
                interval="(lower, upper]",
            )
            for i in range(10)
        ]
    bands = []
    if status == "available":
        band = F.when(missing, "Missing")
        for i, cutoff in enumerate(cutoffs, 1):
            band = band.when(score <= cutoff, f"D{i}")
        aggregated = (
            data.filter(col("cohort").isNotNull())
            .withColumn("band", band.otherwise("D10"))
            .groupBy("cohort", "band")
            .count()
        )
        lookup = {(r["cohort"], r["band"]): r["count"] for r in aggregated.collect()}
        bands = [
            dict(
                cohort=r["cohort"],
                band=b,
                count=lookup.get((r["cohort"], b), 0),
                percentage=100.0 * lookup.get((r["cohort"], b), 0) / r["count"],
            )
            for r in monthly
            for b in BANDS
        ]
    else:
        warnings.append(status)
    in_range = data.filter(valid & score.between(0, 1000))
    bin_id = F.least(F.floor(score / 100).cast("int"), F.lit(9))
    bins = {
        r["bin"]: r["count"]
        for r in in_range.groupBy(bin_id.alias("bin")).count().collect()
    }
    total = sum(bins.values())
    histogram = [
        dict(
            lower=i * 100,
            upper=(i + 1) * 100,
            upper_inclusive=i == 9,
            count=bins.get(i, 0),
            percentage=pct(bins.get(i, 0), total),
            density=bins.get(i, 0) / (total * 100) if total else None,
        )
        for i in range(10)
    ]
    for label, value in {
        **dates,
        **{k: counts[k] for k in ("invalid", "below_range", "above_range")},
    }.items():
        if value:
            warnings.append(f"{label}: {value} received rows.")
    for label, metrics in (("Submitted", sent), ("Received", received)):
        if metrics["null_key_rows"] or metrics["duplicate_key_rows"]:
            warnings.append(
                f"{label}: {metrics['null_key_rows']} null-key rows; "
                f"{metrics['duplicate_key_rows']} duplicate excess rows."
            )
    report = dict(
        matching=matching,
        scores=counts,
        dates=dates,
        monthly_volume=monthly,
        baseline=selected,
        cutoffs=cutoffs,
        band_definitions=definitions,
        monthly_bands=bands,
        histogram=histogram,
        stability_status=status,
        warnings=warnings,
        figures={},
    )
    _render_report(report)
    return report


def _render_report(report):
    """Build optional Matplotlib figures and display ordered notebook sections."""
    try:
        from matplotlib import colormaps
        from matplotlib.figure import Figure
    except ImportError:
        report["warnings"].append("Install matplotlib to generate report figures.")
    else:
        months = [r["cohort"] for r in report["monthly_volume"]]
        for name in ("monthly_volume", "monthly_bands", "histogram"):
            fig = Figure(figsize=(9, 4))
            ax = fig.subplots()
            report["figures"][name] = fig
            if name == "monthly_volume":
                ax.bar(months, [r["count"] for r in report[name]])
                ax.set(
                    xlabel="Monthly cohort",
                    ylabel="Received rows",
                    title="Monthly received volume",
                )
                if not months:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid monthly cohorts",
                        ha="center",
                        transform=ax.transAxes,
                    )
            elif name == "monthly_bands":
                ax.set(
                    xlabel="Monthly cohort",
                    ylabel="Received rows (%)",
                    title="Baseline score-band stability",
                    ylim=(0, 100),
                )
                if report[name]:
                    bottom = [0.0] * len(months)
                    colors = [
                        "#999999",
                        *[colormaps["viridis"](i / 9) for i in range(10)],
                    ]
                    for band, color in zip(BANDS, colors, strict=True):
                        values = [
                            r["percentage"] for r in report[name] if r["band"] == band
                        ]
                        ax.bar(months, values, bottom=bottom, label=band, color=color)
                        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
                    ax.legend(ncol=6, fontsize="small")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        report["stability_status"],
                        ha="center",
                        wrap=True,
                        transform=ax.transAxes,
                    )
            else:
                ax.set(
                    xlabel="Score",
                    ylabel="Probability density (per score point)",
                    title="Overall received score density",
                    xlim=(0, 1000),
                )
                if any(r["count"] for r in report[name]):
                    ax.bar(
                        [r["lower"] for r in report[name]],
                        [r["density"] for r in report[name]],
                        width=100,
                        align="edge",
                        edgecolor="white",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No valid scores in [0, 1000]",
                        ha="center",
                        transform=ax.transAxes,
                    )
            fig.tight_layout()
    try:
        from IPython import get_ipython
        from IPython.display import HTML, Markdown, display
    except ImportError:
        return
    if get_ipython() is None:
        return

    def show_table(rows):
        from html import escape

        if not rows:
            display(Markdown("No rows available."))
            return
        columns = list(rows[0])
        header = "".join(f"<th>{escape(str(c))}</th>" for c in columns)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{escape(str(row.get(c)))}</td>" for c in columns)
            + "</tr>"
            for row in rows
        )
        display(
            HTML(
                f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
            )
        )

    sections = [
        ("Matching summary", "matching"),
        ("Score missingness", "scores"),
        ("Monthly received volume", "monthly_volume"),
        ("Baseline selection and decile thresholds", "band_definitions"),
        ("Monthly score-band stability", "monthly_bands"),
        ("Overall score density", "histogram"),
        ("Data-quality warnings", "warnings"),
    ]
    for title, key in sections:
        display(Markdown(f"### {title}"))
        if key == "band_definitions":
            display(
                {
                    "baseline": report["baseline"],
                    "cutoffs": report["cutoffs"],
                    "status": report["stability_status"],
                }
            )
        if key == "matching":
            show_table(
                [
                    dict(dataset=name, **report[key][name])
                    for name in ("submitted", "received")
                ]
            )
            show_table(
                [
                    dict(metric=k, value=v)
                    for k, v in report[key].items()
                    if k not in ("submitted", "received")
                ]
            )
            display(
                Markdown(
                    "Match percentages use distinct valid keys in each dataset. "
                    "Null-key rows are excluded; duplicates count once for coverage."
                )
            )
        elif key == "scores":
            show_table([report[key]])
            display(Markdown("Missing percentage uses all received rows."))
        elif key == "warnings":
            show_table([{"warning": warning} for warning in report[key]])
        else:
            show_table(report[key])
        if key == "monthly_volume":
            show_table([report["dates"]])
        if key == "monthly_bands":
            display(
                Markdown(
                    "Percentages include all received rows in each valid month. "
                    "Tied scores stay together; baseline bands may differ from 10%."
                )
            )
        if key == "histogram":
            display(
                Markdown(
                    "Density is normalized over finite scores in [0, 1000]. "
                    "Missing, invalid and out-of-range counts are reported above."
                )
            )
        if key in report["figures"]:
            display(report["figures"][key])
