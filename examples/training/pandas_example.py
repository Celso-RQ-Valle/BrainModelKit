"""Run from the repository root after installing .[pandas]."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from brainmodelkit.training.pandas import train_model

X, y = make_classification(n_samples=500, n_features=6, random_state=42)
features = [f"feature_{i}" for i in range(6)]
data = pd.DataFrame(X, columns=features).assign(target=y)
train, oot = train_test_split(data, test_size=0.25, random_state=42, stratify=y)
result = train_model(
    run_name="random_forest_demo",
    target_col="target",
    feature_cols=features,
    train_df=train,
    oot_df=oot,
    model="random_forest",
    model_params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
    df_scoring=oot.drop(columns="target"),
)
print(result.metrics)
print(result.output_dir)
