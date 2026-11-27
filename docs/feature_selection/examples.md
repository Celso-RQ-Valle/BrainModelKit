# Feature selection examples

Choose a function below and run its example after the setup for your backend. Each example returns the selected column names in `result.selected_features`.

## Setup

**pandas** — Run the [pandas data setup](setup.md), then define the training and validation frames used below.

```python
from brainmodelkit.feature_selection import pandas as pd_fs

train, validation = train_df, validation_df
```

**PySpark** — Run the [Spark data setup](spark.md#example-data) to create `train`, `validation`, and `features`, then import the selectors.

```python
from brainmodelkit.feature_selection import pyspark as spark_fs
```

## completeness

[How completeness works](completeness.md#how-it-works).

**pandas**

```python
result = pd_fs.completeness(train, features, min_completeness=0.95)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.completeness(train, features, min_completeness=0.95)
print(result.selected_features)
```

## cardinality

[How cardinality works](cardinality.md#how-it-works).

**pandas**

```python
result = pd_fs.cardinality(train, features, min_unique=2)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.cardinality(train, features, min_unique=2)
print(result.selected_features)
```

## variance_filter

[How variance_filter works](variance_filter.md#how-it-works).

**pandas**

```python
result = pd_fs.variance_filter(train, features, min_variance=0.0)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.variance_filter(train, features, min_variance=0.0)
print(result.selected_features)
```

## correlation_filter

[How correlation_filter works](correlation_filter.md#how-it-works).

**pandas**

```python
result = pd_fs.correlation_filter(train, features, threshold=0.9)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.correlation_filter(train, features, threshold=0.9)
print(result.selected_features)
```

## mutual_information

[How mutual_information works](mutual_information.md#how-it-works).

**pandas**

```python
result = pd_fs.mutual_information(train, "target", features, top_k=1)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.mutual_information(train, "target", features, top_k=1)
print(result.selected_features)
```

## chi_square

[How chi_square works](chi_square.md#how-it-works). Binary indicators satisfy the pandas count and Spark category-code requirements.

**pandas**

```python
counts = counts_df
result = pd_fs.chi_square(counts, "target", features, max_p_value=0.05)
print(result.selected_features)
```

**PySpark**

```python
counts = train.select("target", *((F.col(c) > 0).cast("int").alias(c) for c in features))
result = spark_fs.chi_square(counts, "target", features, max_p_value=0.05)
print(result.selected_features)
```

## anova

[How anova works](anova.md#how-it-works).

**pandas**

```python
result = pd_fs.anova(train, "target", features, max_p_value=0.05)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.anova(train, "target", features, max_p_value=0.05)
print(result.selected_features)
```

## information_value

[How information_value works](information_value.md#how-it-works).

**pandas**

```python
result = pd_fs.information_value(train, "target", features, min_iv=0.02)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.information_value(train, "target", features, min_iv=0.02)
print(result.selected_features)
```

## feature_importance_selection

[How feature_importance_selection works](feature_importance_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.feature_importance_selection(train, "target", features, model="decision_tree", top_k=1)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.feature_importance_selection(train, "target", features, model="decision_tree", top_k=1)
print(result.selected_features)
```

## l1_selection

[How l1_selection works](l1_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.l1_selection(train, "target", features, C=1.0)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.l1_selection(train, "target", features, reg_param=0.1)
print(result.selected_features)
```

## permutation_importance_selection

[How permutation_importance_selection works](permutation_importance_selection.md#how-it-works). Fit on training rows and measure the score change on held-out validation rows.

**pandas**

```python
model = pd_fs.feature_importance_selection(train, "target", features, model="decision_tree").model
result = pd_fs.permutation_importance_selection(model, validation, "target", features, top_k=1)
print(result.selected_features)
```

**PySpark**

```python
model = spark_fs.feature_importance_selection(train, "target", features, model="decision_tree").model
result = spark_fs.permutation_importance_selection(model, validation, "target", features, top_k=1)
print(result.selected_features)
```

## rfe

[How rfe works](rfe.md#how-it-works). Both examples write iteration reports under `feature_selection_runs/`.

**pandas**

```python
result = pd_fs.rfe("example", "target", train, validation, feature_cols=features, n_feat_final=1)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.rfe("example", "target", train, validation, feature_cols=features, n_feat_final=1)
print(result.selected_features)
```

## rfecv

[How rfecv works](rfecv.md#how-it-works).

**pandas**

```python
result = pd_fs.rfecv(train, "target", features, cv=3)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.rfecv(train, "target", features, cv=3, fold_col="fold")
print(result.selected_features)
```

## sequential_selection

[How sequential_selection works](sequential_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.sequential_selection(train, "target", features, n_features_to_select=1, cv=3)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.sequential_selection(train, "target", features, n_features_to_select=1, cv=3, fold_col="fold")
print(result.selected_features)
```

## stability_selection

[How stability_selection works](stability_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.stability_selection(train, "target", features, selector_kwargs={"top_k": 1})
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.stability_selection(train, "target", features, selector_kwargs={"top_k": 1})
print(result.selected_features)
```

## boruta

[How boruta works](boruta.md#what-a-shadow-feature-is). Install the `boruta` extra for the pandas example.

**pandas**

```python
result = pd_fs.boruta(train, "target", features, max_iter=30)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.boruta(train, "target", features, max_iter=30)
print(result.selected_features)
```
