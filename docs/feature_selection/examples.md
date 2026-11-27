# Feature selection examples

Choose a function below and run its example after the setup for your backend. Each example returns the selected column names in `result.selected_features`.

## Find a selector by goal

Choose a goal, follow its example link, and run the [setup](#setup) for your backend. The relative cost tiers apply to both backends. Runtime depends on data size, estimator, and the number of fits or repeats. See the [backend cost comparison](../selection-costs.md) for the main cost drivers.

| Goal | Method and examples | Relative cost |
| --- | --- | --- |
| Remove columns with too many missing values | [`completeness`](#completeness) | Low |
| Drop constant or low-variance columns | [`variance_filter`](#variance_filter) | Low |
| Screen columns by distinct value count | [`cardinality`](#cardinality) | Low to moderate |
| Find numeric features whose class means differ | [`anova`](#anova) | Low to moderate |
| Test counts or category codes for association with the target | [`chi_square`](#chi_square) | Low to moderate |
| Compare event and non-event distributions across bins | [`information_value`](#information_value) | Moderate |
| Detect nonlinear dependence on the target | [`mutual_information`](#mutual_information) | Moderate |
| Remove highly correlated features | [`correlation_filter`](#correlation_filter) | Moderate to high |
| Select a sparse set of logistic regression coefficients | [`l1_selection`](#l1_selection) | Moderate to high |
| Rank features by fitted model importance | [`feature_importance_selection`](#feature_importance_selection) | Moderate to high |
| Measure how much a fitted model relies on each feature | [`permutation_importance_selection`](#permutation_importance_selection) | High |
| Remove the least important features until a chosen size | [`rfe`](#rfe) | High |
| Find features repeatedly selected across samples | [`stability_selection`](#stability_selection) | High to very high |
| Choose the elimination subset size by cross-validation | [`rfecv`](#rfecv) | Very high |
| Build a subset by adding or removing one feature at a time | [`sequential_selection`](#sequential_selection) | Very high |
| Identify all relevant features using shuffled shadow features | [`boruta`](#boruta) | Very high |

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

Each selector call shows every parameter for that backend. Where a selector accepts `df` as an alias for `train_df`, the examples supply `train_df` and leave the alias at `None`. The same convention applies to `oot_df` and its alias `df_oot`. Each method link explains its parameters and selection rules.

## completeness

[How completeness works](completeness.md#how-it-works).

**pandas**

```python
result = pd_fs.completeness(
    train_df=train,
    feature_cols=features,
    df=None,
    min_completeness=0.95,
    group_by=None,
    require_all_groups=False,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.completeness(
    train_df=train,
    feature_cols=features,
    df=None,
    min_completeness=0.95,
    group_by=None,
    require_all_groups=False,
)
print(result.selected_features)
```

## cardinality

[How cardinality works](cardinality.md#how-it-works).

**pandas**

```python
result = pd_fs.cardinality(
    train_df=train,
    feature_cols=features,
    df=None,
    min_unique=2,
    max_unique=None,
    max_unique_ratio=None,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.cardinality(
    train_df=train,
    feature_cols=features,
    df=None,
    min_unique=2,
    max_unique=None,
    max_unique_ratio=None,
)
print(result.selected_features)
```

## variance_filter

[How variance_filter works](variance_filter.md#how-it-works).

**pandas**

```python
result = pd_fs.variance_filter(
    train_df=train,
    feature_cols=features,
    df=None,
    min_variance=0.0,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.variance_filter(
    train_df=train,
    feature_cols=features,
    df=None,
    min_variance=0.0,
)
print(result.selected_features)
```

## correlation_filter

[How correlation_filter works](correlation_filter.md#how-it-works).

**pandas**

```python
result = pd_fs.correlation_filter(
    train_df=train,
    feature_cols=features,
    df=None,
    threshold=0.9,
    method='pearson',
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.correlation_filter(
    train_df=train,
    feature_cols=features,
    df=None,
    threshold=0.9,
    method='pearson',
    max_features=256,
)
print(result.selected_features)
```

## mutual_information

[How mutual_information works](mutual_information.md#how-it-works).

**pandas**

```python
result = pd_fs.mutual_information(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    threshold=None,
    top_k=1,
    random_state=42,
    discrete_features=False,
    n_neighbors=3,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.mutual_information(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    threshold=None,
    top_k=1,
    discrete_features=False,
    n_bins=10,
    relative_error=0.001,
    max_categories=1000,
)
print(result.selected_features)
```

## chi_square

[How chi_square works](chi_square.md#how-it-works). Binary indicators satisfy the pandas count and Spark category-code requirements.

**pandas**

```python
counts = counts_df
result = pd_fs.chi_square(
    train_df=counts,
    target_col='target',
    feature_cols=features,
    df=None,
    max_p_value=0.05,
    top_k=None,
)
print(result.selected_features)
```

**PySpark**

```python
counts = train.select("target", *((F.col(c) > 0).cast("int").alias(c) for c in features))
result = spark_fs.chi_square(
    train_df=counts,
    target_col='target',
    feature_cols=features,
    df=None,
    max_p_value=0.05,
    top_k=None,
    max_categories=1000,
)
print(result.selected_features)
```

## anova

[How anova works](anova.md#how-it-works).

**pandas**

```python
result = pd_fs.anova(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    max_p_value=0.05,
    top_k=None,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.anova(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    max_p_value=0.05,
    top_k=None,
)
print(result.selected_features)
```

## information_value

[How information_value works](information_value.md#how-it-works).

**pandas**

```python
result = pd_fs.information_value(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    n_bins=10,
    min_iv=0.02,
    smoothing=0.5,
    binning='quantile',
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.information_value(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    n_bins=10,
    min_iv=0.02,
    smoothing=0.5,
    binning='quantile',
    relative_error=0.001,
)
print(result.selected_features)
```

## feature_importance_selection

[How feature_importance_selection works](feature_importance_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.feature_importance_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    model='decision_tree',
    model_params=None,
    threshold=None,
    top_k=1,
    random_state=42,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.feature_importance_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    n_tiles=10,
    model='decision_tree',
    model_params=None,
    threshold=None,
    top_k=1,
    random_state=42,
)
print(result.selected_features)
```

## l1_selection

[How l1_selection works](l1_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.l1_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    C=1.0,
    coefficient_tolerance=1e-08,
    random_state=42,
    max_iter=1000,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.l1_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    n_tiles=10,
    reg_param=0.1,
    coefficient_tolerance=1e-08,
    max_iter=100,
)
print(result.selected_features)
```

## permutation_importance_selection

[How permutation_importance_selection works](permutation_importance_selection.md#how-it-works). Fit on training rows and measure the score change on held-out validation rows.

**pandas**

```python
model = pd_fs.feature_importance_selection(train, "target", features, model="decision_tree").model
result = pd_fs.permutation_importance_selection(
    model=model,
    df=validation,
    target_col='target',
    feature_cols=features,
    scoring='roc_auc',
    n_repeats=5,
    random_state=42,
    threshold=None,
    top_k=1,
    n_jobs=None,
)
print(result.selected_features)
```

**PySpark**

```python
model = spark_fs.feature_importance_selection(train, "target", features, model="decision_tree").model
result = spark_fs.permutation_importance_selection(
    model=model,
    df=validation,
    target_col='target',
    feature_cols=features,
    scoring='roc_auc',
    n_repeats=5,
    random_state=42,
    threshold=None,
    top_k=1,
)
print(result.selected_features)
```

## rfe

[How rfe works](rfe.md#how-it-works). Both examples write iteration reports under `feature_selection_runs/`.

**pandas**

```python
result = pd_fs.rfe(
    run_name='example',
    target_col='target',
    train_df=train,
    oot_df=validation,
    feature_cols=features,
    model='logistic_regression',
    df_oot=None,
    step=1,
    n_feat_final=1,
    model_params=None,
    output_dir='feature_selection_runs',
    df_scoring=None,
    mlflow_logging=False,
    signature=False,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.rfe(
    run_name='example',
    target_col='target',
    train_df=train,
    oot_df=validation,
    feature_cols=features,
    model='logistic_regression',
    df_oot=None,
    step=1,
    n_feat_final=1,
    model_params=None,
    output_dir='feature_selection_runs',
    df_scoring=None,
    mlflow_logging=False,
    signature=False,
    n_tiles=10,
)
print(result.selected_features)
```

## rfecv

[How rfecv works](rfecv.md#how-it-works).

**pandas**

```python
result = pd_fs.rfecv(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    model='logistic_regression',
    model_params=None,
    scoring='roc_auc',
    cv=3,
    step=1,
    min_features_to_select=1,
    n_jobs=None,
    random_state=42,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.rfecv(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    model='logistic_regression',
    model_params=None,
    scoring='roc_auc',
    cv=3,
    fold_col='fold',
    step=1,
    min_features_to_select=1,
    random_state=42,
)
print(result.selected_features)
```

## sequential_selection

[How sequential_selection works](sequential_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.sequential_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    model='logistic_regression',
    model_params=None,
    direction='forward',
    n_features_to_select=1,
    scoring='roc_auc',
    cv=3,
    n_jobs=None,
    random_state=42,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.sequential_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    model='logistic_regression',
    model_params=None,
    direction='forward',
    n_features_to_select=1,
    scoring='roc_auc',
    cv=3,
    fold_col='fold',
    random_state=42,
)
print(result.selected_features)
```

## stability_selection

[How stability_selection works](stability_selection.md#how-it-works).

**pandas**

```python
result = pd_fs.stability_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    method='feature_importance',
    n_iterations=20,
    sample_fraction=0.8,
    min_frequency=0.8,
    random_state=42,
    group_by=None,
    selector_kwargs={'top_k': 1},
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.stability_selection(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    method='feature_importance',
    n_iterations=20,
    sample_fraction=0.8,
    min_frequency=0.8,
    random_state=42,
    group_by=None,
    selector_kwargs={'top_k': 1},
    max_groups=100,
)
print(result.selected_features)
```

## boruta

[How boruta works](boruta.md#what-a-shadow-feature-is). Install the `boruta` extra for the pandas example.

**pandas**

```python
result = pd_fs.boruta(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    oot_df=None,
    df_oot=None,
    df_scoring=None,
    run_name='feature_selection',
    model='random_forest',
    model_params=None,
    n_estimators='auto',
    max_iter=30,
    perc=100,
    alpha=0.05,
    random_state=42,
)
print(result.selected_features)
```

**PySpark**

```python
result = spark_fs.boruta(
    train_df=train,
    target_col='target',
    feature_cols=features,
    df=None,
    model='random_forest',
    model_params=None,
    n_estimators=100,
    max_iter=30,
    perc=100,
    alpha=0.05,
    two_step=True,
    random_state=42,
)
print(result.selected_features)
```
