"""Execute Pandas RFE with a generated binary dataset."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from brainmodelkit.feature_selection.pandas import rfe

X, y = make_classification(n_samples=500, n_features=6, random_state=42)
features = [f"feat_{i}" for i in range(6)]
data = pd.DataFrame(X, columns=features).assign(target=y)
train, oot = train_test_split(data, test_size=0.25, random_state=42, stratify=y)
selection = rfe(
    "pandas_random_forest_rfe",
    "target",
    train,
    oot,
    feature_cols=features,
    model="random_forest",
    model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
    step=2,
    n_feat_final=3,
)
print(selection.selected_features)
print(selection.training_result.metrics)
print(selection.output_dir)
