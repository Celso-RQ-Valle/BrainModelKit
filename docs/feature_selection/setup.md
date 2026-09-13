# Shared setup for method examples

The per-method snippets use these names. Run this setup once in a notebook,
then run the snippet for the chosen method. Install `BrainModelKit[pandas]`;
add the `boruta` extra for Pandas Boruta. For a standalone executable example use
[method_example.py](../../examples/feature_selection/method_example.py).

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from brainmodelkit.feature_selection import pandas as fs

x, y = make_classification(
    n_samples=500, n_features=6, n_informative=3, n_redundant=0, random_state=42
)
features = [f"feat_{i}" for i in range(6)]
data = pd.DataFrame(x, columns=features).assign(target=y)
train_df, oot_df = train_test_split(data, stratify=y, random_state=42, test_size=0.25)
train_df = train_df.assign(reference_date="development")
df = train_df
validation_df = oot_df
counts_df = df.copy()
counts_df[features] = (df[features] > 0).astype(int)
count_features = features
model = fs.feature_importance_selection(df, "target", features).model
fitted_model = model
```

The numeric columns are finite and require no imputation for this synthetic demo.
For real data, fit preprocessing on training rows and respect validation fold
boundaries. `reference_date` here is a display group, not a realistic time split.
The [Spark setup](spark.md#example-data) creates equivalent demonstration variables
directly with Spark expressions.
