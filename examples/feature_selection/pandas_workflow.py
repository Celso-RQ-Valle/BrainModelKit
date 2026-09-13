"""Compose Pandas filters and existing RFE on a reproducible development split."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from brainmodelkit.feature_selection.pandas import (
    completeness,
    correlation_filter,
    information_value,
    rfe,
)

x, y = make_classification(n_samples=300, n_features=6, random_state=42)
features = [f"feat_{i}" for i in range(6)]
df = pd.DataFrame(x, columns=features).assign(target=y)
train, oot = train_test_split(df, test_size=0.25, stratify=y, random_state=42)

features = completeness(train, features, min_completeness=0.70).selected_features
features = correlation_filter(train, features, threshold=0.90).selected_features
iv = information_value(train, "target", features, min_iv=0.02)
features = iv.selected_features
print(iv.feature_table)
if not features:
    raise ValueError("No features passed the screening thresholds")
result = rfe(
    "pandas_screened_rfe",
    "target",
    train,
    oot,
    feature_cols=features,
    n_feat_final=min(2, len(features)),
)
print(result.selected_features)
print(result.training_result.metrics)
