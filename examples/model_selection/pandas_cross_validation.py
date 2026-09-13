"""Small Pandas CV example."""

import pandas as pd
from sklearn.datasets import make_classification

from brainmodelkit.model_selection.pandas import cross_validate

X, y = make_classification(n_samples=100, n_features=3, n_redundant=0, random_state=42)
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
df["target"] = y
result = cross_validate(df, "target", ["x1", "x2", "x3"], "logistic_regression")
print(result.fold_metrics)
print(result.summary)
