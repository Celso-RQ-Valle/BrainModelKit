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
COMPANY_SIZES = ("Small", "Medium", "Large", "Very Large")

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
COMPANY_SIZE_RISK_ADJUSTMENT = {
    "Small": 0.08,
    "Medium": 0.03,
    "Large": -0.03,
    "Very Large": -0.07,
}


def _generate_company_id(generator: random.Random) -> str:
    digits = [generator.randint(0, 9) for _ in range(14)]
    return (
        f"{digits[0]}{digits[1]}.{digits[2]}{digits[3]}{digits[4]}."
        f"{digits[5]}{digits[6]}{digits[7]}/0001-{digits[12]}{digits[13]}"
    )


def _generate_reference_date(generator: random.Random, max_days: int) -> date:
    return datetime.today().date() + timedelta(days=generator.randint(1, max_days))


def _default_probability(
    status: str,
    industry_section: str,
    company_size: str,
) -> float:
    probability = INDUSTRY_RISK.get(industry_section, 0.40)
    probability += STATUS_RISK_ADJUSTMENT.get(status, 0.0)
    probability += COMPANY_SIZE_RISK_ADJUSTMENT.get(company_size, 0.0)
    return min(max(probability, 0.01), 0.99)


def _generate_features(
    generator: random.Random,
    default_flag: int,
    feature_count: int,
    informative_indices: set[int],
    signal_directions: dict[int, int],
    signal_strength: float,
) -> tuple[float, ...]:
    """Generate noisy features, with moderate signal in selected columns.

    The baseline columns are independent standard-noise variables. Informative
    columns add a noisy, randomly oriented target effect; the noise is larger
    than the effect so the signal is useful for demonstrations without being a
    direct copy of the target.
    """
    features = []
    for index in range(feature_count):
        value = generator.gauss(0, 1)
        if index in informative_indices:
            value += signal_directions[index] * (default_flag - 0.5) * signal_strength
        features.append(round(value, 6))
    return tuple(features)


def simulate_credit_data(
    spark_session: SparkSession | None = None,
    *,
    row_count: int = 10_000,
    feature_count: int = 20,
    max_days: int = 30,
    seed: int = 42,
    informative_fraction: float = 1 / 6,
    signal_strength: float = 0.75,
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
    informative_fraction:
        Fraction of features with a weak target-related signal. Selected
        columns are chosen reproducibly from the seed; at least one is used
        when the fraction is greater than zero. The default is approximately
        one in six features.
    signal_strength:
        Strength of the noisy target effect. It is deliberately smaller than
        the feature noise scale by default to avoid trivially predictive data.
    """
    if row_count < 1:
        raise ValueError("row_count must be at least 1")
    if feature_count < 1:
        raise ValueError("feature_count must be at least 1")
    if max_days < 1:
        raise ValueError("max_days must be at least 1")
    if not 0 <= informative_fraction <= 1:
        raise ValueError("informative_fraction must be between 0 and 1")
    if signal_strength <= 0:
        raise ValueError("signal_strength must be greater than 0")

    if spark_session is None:
        try:
            from pyspark.sql import SparkSession
        except ImportError as error:
            raise ImportError(
                "PySpark is required. Install it with "
                "`pip install 'BrainModelKit[pyspark]'`."
            ) from error
        spark_session = SparkSession.builder.appName(
            "SyntheticCreditData"
        ).getOrCreate()

    generator = random.Random(seed)
    rows: list[tuple[Any, ...]] = []
    informative_count = (
        max(1, int(feature_count * informative_fraction))
        if informative_fraction > 0
        else 0
    )
    informative_indices = set(generator.sample(range(feature_count), informative_count))
    signal_directions = {
        index: generator.choice((-1, 1)) for index in informative_indices
    }

    for _ in range(row_count):
        status = generator.choice(STATUS_OPTIONS)
        industry_section = generator.choice(INDUSTRY_SECTIONS)
        company_size = generator.choice(COMPANY_SIZES)
        probability = _default_probability(
            status,
            industry_section,
            company_size,
        )
        default_flag = int(generator.random() < probability)
        features = _generate_features(
            generator,
            default_flag,
            feature_count,
            informative_indices,
            signal_directions,
            signal_strength,
        )
        rows.append(
            (
                _generate_company_id(generator),
                _generate_reference_date(generator, max_days),
                status,
                default_flag,
                industry_section,
                company_size,
                *features,
            )
        )

    feature_columns = [f"feature_{index:02d}" for index in range(1, feature_count + 1)]
    columns = [
        "company_id",
        "reference_date",
        "status",
        "default_flag",
        "industry_section",
        "company_size",
        *feature_columns,
    ]
    return spark_session.createDataFrame(rows, columns), feature_columns


__all__ = ["simulate_credit_data"]
