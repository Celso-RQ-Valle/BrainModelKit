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

**PySpark** — Run the [Spark data setup](spark.md#example-data) to create `train`, `validation`, and `features`, then import the selectors.

```python
from brainmodelkit.feature_selection import pyspark as spark_fs
```

**pandas** — Run the [pandas data setup](setup.md), then define the training and validation frames used below.

```python
from brainmodelkit.feature_selection import pandas as pd_fs

train, validation = train_df, validation_df
```

Each selector call shows every parameter for that backend. Single-frame selectors take `df`. Selectors with OOT evaluation take `df_train` and `df_oot`. Each method link explains its parameters and selection rules.

For completeness and stability selection, pass grouping columns as a list, even for one column. Use `None` for the whole training frame. With two or more columns, each distinct combination of their values defines a group.

## completeness

[How completeness works](completeness.md#how-it-works).

**PySpark**

```python
result = spark_fs.completeness(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_completeness=0.95,  # Keep columns with observed values in at least this fraction of rows.
    group_by=["fold"],  # Report completeness for each observed group.
    require_all_groups=False,  # Require the completeness cutoff in every group when enabled.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.completeness(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_completeness=0.95,  # Keep columns with observed values in at least this fraction of rows.
    group_by=["reference_date"],  # Report completeness for each observed group.
    require_all_groups=False,  # Require the completeness cutoff in every group when enabled.
)
print(result.selected_features)
```

## cardinality

[How cardinality works](cardinality.md#how-it-works).

**PySpark**

```python
result = spark_fs.cardinality(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_unique=2,  # Minimum number of distinct nonmissing values to keep a column.
    max_unique=None,  # Optional upper limit on distinct nonmissing values.
    max_unique_ratio=None,  # Optional upper limit on distinct nonmissing values divided by total rows.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.cardinality(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_unique=2,  # Minimum number of distinct nonmissing values to keep a column.
    max_unique=None,  # Optional upper limit on distinct nonmissing values.
    max_unique_ratio=None,  # Optional upper limit on distinct nonmissing values divided by total rows.
)
print(result.selected_features)
```

## variance_filter

[How variance_filter works](variance_filter.md#how-it-works).

**PySpark**

```python
result = spark_fs.variance_filter(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_variance=0.0,  # Keep columns whose population variance exceeds this cutoff.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.variance_filter(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    min_variance=0.0,  # Keep columns whose population variance exceeds this cutoff.
)
print(result.selected_features)
```

## correlation_filter

[How correlation_filter works](correlation_filter.md#how-it-works).

**PySpark**

```python
result = spark_fs.correlation_filter(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    threshold=0.9,  # Reject a later column if its absolute correlation with a retained one reaches this value.
    method="pearson",  # Correlation measure used to compare feature pairs.
    max_features=256,  # Limit the number of columns in the driver-side correlation matrix.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.correlation_filter(
    df=train,  # Training rows used to fit the selector.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    threshold=0.9,  # Reject a later column if its absolute correlation with a retained one reaches this value.
    method="pearson",  # Correlation measure used to compare feature pairs.
)
print(result.selected_features)
```

## mutual_information

[How mutual_information works](mutual_information.md#how-it-works).

**PySpark**

```python
result = spark_fs.mutual_information(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=1,  # Optional limit on the number of highest-ranked features retained.
    discrete_features=False,  # Identify discrete inputs. False treats the feature columns as continuous.
    n_bins=10,  # Requested number of bins for continuous values.
    relative_error=0.001,  # Allowed rank error when Spark approximates quantile boundaries.
    max_categories=1000,  # Limit distinct levels per discrete feature to bound counting costs.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.mutual_information(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=1,  # Optional limit on the number of highest-ranked features retained.
    random_state=42,  # Seed for reproducible random choices.
    discrete_features=False,  # Identify discrete inputs. False treats the feature columns as continuous.
    n_neighbors=3,  # Number of neighbors used to estimate dependence for continuous inputs.
)
print(result.selected_features)
```

## chi_square

[How chi_square works](chi_square.md#how-it-works). Binary indicators satisfy the pandas count and Spark category-code requirements.

**PySpark**

```python
counts = train.select(
    "target", *((F.col(c) > 0).cast("int").alias(c) for c in features)
)
result = spark_fs.chi_square(
    df=counts,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    max_p_value=0.05,  # Largest unadjusted p-value that qualifies a feature for selection.
    top_k=None,  # Optional limit on the number of highest-ranked features retained.
    max_categories=1000,  # Limit distinct levels per discrete feature to bound counting costs.
)
print(result.selected_features)
```

**pandas**

```python
counts = counts_df
result = pd_fs.chi_square(
    df=counts,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    max_p_value=0.05,  # Largest unadjusted p-value that qualifies a feature for selection.
    top_k=None,  # Optional limit on the number of highest-ranked features retained.
)
print(result.selected_features)
```

## anova

[How anova works](anova.md#how-it-works).

**PySpark**

```python
result = spark_fs.anova(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    max_p_value=0.05,  # Largest unadjusted p-value that qualifies a feature for selection.
    top_k=None,  # Optional limit on the number of highest-ranked features retained.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.anova(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    max_p_value=0.05,  # Largest unadjusted p-value that qualifies a feature for selection.
    top_k=None,  # Optional limit on the number of highest-ranked features retained.
)
print(result.selected_features)
```

## information_value

[How information_value works](information_value.md#how-it-works).

**PySpark**

```python
result = spark_fs.information_value(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    n_bins=10,  # Requested number of bins for continuous values.
    min_iv=0.02,  # Minimum information value required to retain a feature.
    smoothing=0.5,  # Pseudocount added to event and non-event bin counts before calculating WoE.
    binning="quantile",  # Rule for placing numeric bin boundaries. Quantile uses sample quantiles.
    relative_error=0.001,  # Allowed rank error when Spark approximates quantile boundaries.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.information_value(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    n_bins=10,  # Requested number of bins for continuous values.
    min_iv=0.02,  # Minimum information value required to retain a feature.
    smoothing=0.5,  # Pseudocount added to event and non-event bin counts before calculating WoE.
    binning="quantile",  # Rule for placing numeric bin boundaries. Quantile uses sample quantiles.
)
print(result.selected_features)
```

## feature_importance_selection

[How feature_importance_selection works](feature_importance_selection.md#how-it-works).

**PySpark**

```python
result = spark_fs.feature_importance_selection(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    n_tiles=10,  # Number of score tiles used for Spark evaluation metrics.
    model="decision_tree",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=1,  # Optional limit on the number of highest-ranked features retained.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.feature_importance_selection(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    model="decision_tree",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=1,  # Optional limit on the number of highest-ranked features retained.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

## l1_selection

[How l1_selection works](l1_selection.md#how-it-works).

**PySpark**

```python
result = spark_fs.l1_selection(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    n_tiles=10,  # Number of score tiles used for Spark evaluation metrics.
    reg_param=0.1,  # Regularization strength. Larger values shrink coefficients more.
    coefficient_tolerance=1e-08,  # Keep features whose absolute coefficient exceeds this value.
    max_iter=100,  # Maximum solver iterations for the regularized logistic fit.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.l1_selection(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    C=1.0,  # Inverse regularization strength. Smaller values shrink coefficients more.
    coefficient_tolerance=1e-08,  # Keep features whose absolute coefficient exceeds this value.
    random_state=42,  # Seed for reproducible random choices.
    max_iter=1000,  # Maximum solver iterations for the regularized logistic fit.
)
print(result.selected_features)
```

## permutation_importance_selection

[How permutation_importance_selection works](permutation_importance_selection.md#how-it-works). Fit on training rows and measure the score change on held-out validation rows.

**PySpark**

```python
model = spark_fs.feature_importance_selection(
    train, "target", features, model="decision_tree"
).model
result = spark_fs.permutation_importance_selection(
    model=model,  # Already-fitted classifier to rescore after each shuffle.
    df=validation,  # Held-out rows on which shuffling a feature measures the score decrease.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    n_repeats=5,  # Number of shuffles per feature when measuring the score decrease.
    random_state=42,  # Seed for reproducible random choices.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=1,  # Optional limit on the number of highest-ranked features retained.
)
print(result.selected_features)
```

**pandas**

```python
model = pd_fs.feature_importance_selection(
    train, "target", features, model="decision_tree"
).model
result = pd_fs.permutation_importance_selection(
    model=model,  # Already-fitted classifier to rescore after each shuffle.
    df=validation,  # Held-out rows on which shuffling a feature measures the score decrease.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    n_repeats=5,  # Number of shuffles per feature when measuring the score decrease.
    random_state=42,  # Seed for reproducible random choices.
    threshold=None,  # Optional minimum score a feature must reach to qualify.
    top_k=1,  # Optional limit on the number of highest-ranked features retained.
    n_jobs=None,  # Number of parallel workers. None uses the configured joblib default.
)
print(result.selected_features)
```

## rfe

[How rfe works](rfe.md#how-it-works). Both examples write iteration reports under `feature_selection_runs/`.

**PySpark**

```python
result = spark_fs.rfe(
    run_name="example",  # Label used to identify the training run and its outputs.
    target_col="target",  # Column containing the class label.
    df_train=train,  # Training rows used to fit the selector.
    df_oot=validation,  # Held-out rows used to evaluate the elimination fits.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    step=1,  # Number of features removed in each elimination round.
    n_feat_final=1,  # Number of features to retain at the end of elimination.
    model_params=None,  # Estimator settings.
    output_dir="feature_selection_runs",  # Parent directory for iteration reports and selected-feature outputs.
    df_scoring=None,  # Optional additional frame to score with the final model.
    mlflow_logging=False,  # Enable MLflow tracking for this run.
    signature=False,  # Include an MLflow model signature when logging the model.
    n_tiles=10,  # Number of score tiles used for Spark evaluation metrics.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.rfe(
    run_name="example",  # Label used to identify the training run and its outputs.
    target_col="target",  # Column containing the class label.
    df_train=train,  # Training rows used to fit the selector.
    df_oot=validation,  # Held-out rows used to evaluate the elimination fits.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    step=1,  # Number of features removed in each elimination round.
    n_feat_final=1,  # Number of features to retain at the end of elimination.
    model_params=None,  # Estimator settings.
    output_dir="feature_selection_runs",  # Parent directory for iteration reports and selected-feature outputs.
    df_scoring=None,  # Optional additional frame to score with the final model.
    mlflow_logging=False,  # Enable MLflow tracking for this run.
    signature=False,  # Include an MLflow model signature when logging the model.
)
print(result.selected_features)
```

## rfecv

[How rfecv works](rfecv.md#how-it-works).

**PySpark**

```python
result = spark_fs.rfecv(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=3,  # Number of cross-validation folds used to score candidate subsets.
    fold_col="fold",  # Column assigning each training row to a validation fold.
    step=1,  # Number of features removed in each elimination round.
    min_features_to_select=1,  # Smallest subset size considered during elimination.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.rfecv(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=3,  # Number of cross-validation folds used to score candidate subsets.
    step=1,  # Number of features removed in each elimination round.
    min_features_to_select=1,  # Smallest subset size considered during elimination.
    n_jobs=None,  # Number of parallel workers. None uses the configured joblib default.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

## sequential_selection

[How sequential_selection works](sequential_selection.md#how-it-works).

**PySpark**

```python
result = spark_fs.sequential_selection(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    direction="forward",  # Forward adds one feature at a time. Backward starts with all candidates.
    n_features_to_select=1,  # Number of features in the final greedy subset.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=3,  # Number of cross-validation folds used to score candidate subsets.
    fold_col="fold",  # Column assigning each training row to a validation fold.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.sequential_selection(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    model="logistic_regression",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    direction="forward",  # Forward adds one feature at a time. Backward starts with all candidates.
    n_features_to_select=1,  # Number of features in the final greedy subset.
    scoring="roc_auc",  # Metric used to evaluate predictions. Here it is area under the ROC curve.
    cv=3,  # Number of cross-validation folds used to score candidate subsets.
    n_jobs=None,  # Number of parallel workers. None uses the configured joblib default.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

## stability_selection

[How stability_selection works](stability_selection.md#how-it-works).

**PySpark**

```python
result = spark_fs.stability_selection(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    method="feature_importance",  # Base selector to run on each resample or group.
    n_iterations=20,  # Number of bootstrap selection rounds.
    sample_fraction=0.8,  # Size of each bootstrap sample relative to the training data.
    min_frequency=0.8,  # Minimum fraction of selections a feature must appear in to survive.
    random_state=42,  # Seed for reproducible random choices.
    group_by=["fold"],  # Resample within each group.
    selector_kwargs={"top_k": 1},  # Base selector options.
    max_groups=100,  # Maximum number of groups Spark will process for grouped selection.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.stability_selection(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    method="feature_importance",  # Base selector to run on each resample or group.
    n_iterations=20,  # Number of bootstrap selection rounds.
    sample_fraction=0.8,  # Size of each bootstrap sample relative to the training data.
    min_frequency=0.8,  # Minimum fraction of selections a feature must appear in to survive.
    random_state=42,  # Seed for reproducible random choices.
    group_by=["reference_date"],  # Resample within each group.
    selector_kwargs={"top_k": 1},  # Base selector options.
)
print(result.selected_features)
```

## boruta

[How boruta works](boruta.md#what-a-shadow-feature-is). Install the `boruta` extra for the pandas example.

**PySpark**

```python
result = spark_fs.boruta(
    df=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    model="random_forest",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    n_estimators=100,  # Number of trees used to compare original and shadow features.
    max_iter=30,  # Iteration budget for comparing features with randomized shadows.
    perc=100,  # Percentile of shadow importances that an original feature must beat.
    alpha=0.05,  # Significance level for testing repeated wins against shadow features.
    two_step=True,  # Apply the two-stage multiple-testing correction for Boruta decisions.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```

**pandas**

```python
result = pd_fs.boruta(
    df_train=train,  # Training rows used to fit the selector.
    target_col="target",  # Column containing the class label.
    feature_cols=features,  # Ordered candidate columns to evaluate.
    df_oot=None,  # Optional held-out frame for evaluating the selected-feature model.
    df_scoring=None,  # Optional additional frame to score with the final model.
    run_name="feature_selection",  # Label used to identify the training run and its outputs.
    model="random_forest",  # Estimator used to fit the candidate features.
    model_params=None,  # Estimator settings.
    n_estimators="auto",  # Let BorutaPy choose the forest size from the remaining feature count.
    max_iter=30,  # Iteration budget for comparing features with randomized shadows.
    perc=100,  # Percentile of shadow importances that an original feature must beat.
    alpha=0.05,  # Significance level for testing repeated wins against shadow features.
    random_state=42,  # Seed for reproducible random choices.
)
print(result.selected_features)
```
