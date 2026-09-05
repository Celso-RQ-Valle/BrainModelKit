"""Synthetic credit data generation for PySpark workflows."""

from __future__ import annotations

import os
import random
import sys
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

# Keep Spark workers on the active interpreter, especially in Windows notebooks.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

STATUS_OPTIONS = (
    "Approved and Contracted",
    "Approved Not Contracted",
    "Rejected",
)
INDUSTRY_SECTIONS = tuple("ABCDEFGHIJKLMNOPQRSTU")

INDUSTRY_RISK = {
    "A": 0.65,
    "B": 0.60,
    "C": 0.55,
    "D": 0.50,
    "E": 0.48,
    "F": 0.45,
    "G": 0.42,
    "H": 0.40,
    "I": 0.38,
    "J": 0.36,
    "K": 0.34,
    "L": 0.32,
    "M": 0.30,
    "N": 0.28,
    "O": 0.26,
    "P": 0.24,
    "Q": 0.22,
    "R": 0.20,
    "S": 0.18,
    "T": 0.16,
    "U": 0.14,
}
STATUS_RISK_ADJUSTMENT = {
    "Approved and Contracted": -0.15,
    "Approved Not Contracted": -0.05,
    "Rejected": 0.10,
}


def _generate_company_id(generator: random.Random) -> str:
    digits = [generator.randint(0, 9) for _ in range(14)]
    return (
        f"{digits[0]}{digits[1]}.{digits[2]}{digits[3]}{digits[4]}."
        f"{digits[5]}{digits[6]}{digits[7]}/0001-{digits[12]}{digits[13]}"
    )


def _generate_reference_date(generator: random.Random, max_days: int) -> date:
    return datetime.today().date() + timedelta(days=generator.randint(1, max_days))


def _default_probability(status: str, industry_section: str) -> float:
    probability = INDUSTRY_RISK.get(industry_section, 0.40)
    probability += STATUS_RISK_ADJUSTMENT.get(status, 0.0)
    return min(max(probability, 0.01), 0.99)


def _generate_features(
    generator: random.Random,
    default_probability: float,
    feature_count: int,
) -> tuple[float, ...]:
    return tuple(
        round(
            default_probability * (1 + index / feature_count)
            + generator.gauss(0, 0.15),
            6,
        )
        for index in range(feature_count)
    )


def simulate_credit_data(
    spark_session: SparkSession | None = None,
    *,
    row_count: int = 10_000,
    feature_count: int = 20,
    max_days: int = 30,
    seed: int = 42,
) -> tuple[DataFrame, list[str]]:
    """Create a synthetic credit DataFrame and return its numeric feature names.

    Parameters
    ----------
    spark_session:
        Existing Spark session. When omitted, a local session is obtained or created.
    row_count:
        Number of synthetic credit records to generate.
    feature_count:
        Number of numeric model features. Defaults to 20.
    max_days:
        Maximum number of days after today for the reference date.
    seed:
        Random seed used to make generated values reproducible.
    """
    if row_count < 1:
        raise ValueError("row_count must be at least 1")
    if feature_count < 1:
        raise ValueError("feature_count must be at least 1")
    if max_days < 1:
        raise ValueError("max_days must be at least 1")

    if spark_session is None:
        try:
            from pyspark.sql import SparkSession
        except ImportError as error:
            raise ImportError(
                "PySpark is required. Install it with "
                "`pip install 'BrainModelKit[pyspark]'`."
            ) from error
        spark_session = (
            SparkSession.builder.appName("SyntheticCreditData").getOrCreate()
        )

    generator = random.Random(seed)
    rows: list[tuple[Any, ...]] = []

    for _ in range(row_count):
        status = generator.choice(STATUS_OPTIONS)
        industry_section = generator.choice(INDUSTRY_SECTIONS)
        probability = _default_probability(status, industry_section)
        features = _generate_features(
            generator,
            probability,
            feature_count,
        )
        rows.append(
            (
                _generate_company_id(generator),
                _generate_reference_date(generator, max_days),
                status,
                int(generator.random() < probability),
                industry_section,
                *features,
            )
        )

    feature_columns = [
        f"feature_{index:02d}" for index in range(1, feature_count + 1)
    ]
    columns = [
        "company_id",
        "reference_date",
        "status",
        "default_flag",
        "industry_section",
        *feature_columns,
    ]
    return spark_session.createDataFrame(rows, columns), feature_columns


__all__ = ["simulate_credit_data"]
