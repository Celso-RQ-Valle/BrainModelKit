"""Native Spark subset search and permutation; no local dataset replicas.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

from contextlib import contextmanager
from uuid import uuid4

import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StructField, StructType

from brainmodelkit.model_selection.pyspark import cross_validate

from ._selection import SelectionResult, integer, ranked, resolve_frame
from ._spark_model import feature_importance_selection
from ._spark_quality import col
from ._spark_statistics import supervised, vector_frame


def score(model, frame, target, features, scoring):
    names = {"roc_auc": "areaUnderROC", "average_precision": "areaUnderPR"}
    if scoring not in names:
        raise ValueError("scoring must be roc_auc or average_precision (Spark PR area)")
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    value = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol=model.getRawPredictionCol(),
        metricName=names[scoring],
        numBins=0,
    ).evaluate(model.transform(vector_frame(frame, features, target)))
    if not np.isfinite(value):
        raise ValueError("Evaluation score must be finite")
    return float(value)


@contextmanager
def owned_cache(frame):
    """Materialize an isolated cache and release it even when an action fails.

    The unique literal prevents cache-plan matching from taking ownership of a
    caller's cached query. Unlike localCheckpoint, unpersist releases this cache.
    """
    token = uuid4().hex
    marker = "__bmk_cache_" + token
    cached = frame.withColumn(marker, F.lit(token)).persist()
    try:
        cached.count()
        yield cached.drop(marker)
    finally:
        cached.unpersist(blocking=True)


@contextmanager
def indexed(frame):
    """Own and release an indexed cache, including duplicate source rows."""
    if "__bmk_id" in frame.columns:
        raise ValueError("Column __bmk_id is reserved for permutation")
    schema = StructType(
        [*frame.schema.fields, StructField("__bmk_id", LongType(), False)]
    )
    result = frame.sparkSession.createDataFrame(
        frame.rdd.zipWithIndex().map(lambda pair: (*pair[0], pair[1])), schema
    )
    with owned_cache(result) as cached:
        yield cached


def shuffled(frame, feature, seed, output=None):
    """Random distributed sort and positional join give a true permutation."""
    output = output or feature
    schema = StructType(
        [
            StructField(output, frame.schema[feature].dataType, True),
            StructField("__bmk_id", LongType(), False),
        ]
    )
    values = frame.select(col(feature)).orderBy(F.rand(seed))
    replacement = frame.sparkSession.createDataFrame(
        values.rdd.zipWithIndex().map(lambda pair: (pair[0][0], pair[1])), schema
    )
    base = frame.drop(feature) if output == feature else frame
    return base.join(replacement, "__bmk_id")


def permutation_importance_selection(
    model,
    df,
    target_col,
    feature_cols,
    *,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    threshold=None,
    top_k=None,
):
    """Rank held-out score decreases after exact distributed column permutations.

    The fitted native classifier must expect the vector assembled in feature_cols
    order. See docs/feature_selection/spark.md#permutation_importance_selection.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.permutation_importance_selection(
            model, validation_df, "target", ["income", "age"]
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/permutation_importance_selection.md
    """
    features = supervised(df, target_col, feature_cols)
    integer(n_repeats, "n_repeats")
    rows = []
    with indexed(df.select(*[col(f) for f in [*features, target_col]])) as frame:
        baseline = score(model, frame, target_col, features, scoring)
        for i, feature in enumerate(features):
            values = [
                baseline
                - score(
                    model,
                    shuffled(frame, feature, random_state + i * n_repeats + j),
                    target_col,
                    features,
                    scoring,
                )
                for j in range(n_repeats)
            ]
            rows.append(
                {
                    "feature": feature,
                    "importance_mean": float(np.mean(values)),
                    "importance_std": float(np.std(values)),
                }
            )
    result = ranked(rows, "importance_mean", "permutation_importance", threshold, top_k)
    result.metadata.update(
        baseline_score=baseline, scoring=scoring, n_repeats=n_repeats
    )
    return result


def fit(frame, target, features, model, params, seed):
    return feature_importance_selection(
        frame, target, features, model=model, model_params=params, random_state=seed
    )


def path(frame, target, features, model, params, seed, step, minimum):
    current = list(features)
    while True:
        result = fit(frame, target, current, model, params, seed)
        yield current.copy(), result
        if len(current) == minimum:
            break
        values = {r["feature"]: abs(r["importance"]) for r in result.feature_table}
        removed = sorted(current, key=values.__getitem__)[
            : min(step, len(current) - minimum)
        ]
        current = [f for f in current if f not in removed]


def rfecv(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    model="logistic_regression",
    model_params=None,
    scoring="roc_auc",
    cv=5,
    fold_col=None,
    step=1,
    min_features_to_select=1,
    random_state=42,
):
    """Choose feature count from fold-local elimination paths; refit on all training.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.rfecv(
            train_df, "target", ["income", "age"], cv=3
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/rfecv.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    integer(step, "step")
    integer(min_features_to_select, "min_features_to_select")
    if min_features_to_select > len(features):
        raise ValueError("min_features_to_select exceeds feature count")
    if scoring != "roc_auc":
        raise ValueError("rfecv uses Spark cross_validate metrics and requires roc_auc")
    scores = {}
    subsets = []
    for subset, _ in path(
        train_df,
        target_col,
        features,
        model,
        model_params,
        random_state,
        step,
        min_features_to_select,
    ):
        subsets.append(subset)
        cv_result = cross_validate(
            train_df,
            target_col,
            subset,
            model,
            model_params=model_params,
            cv=cv,
            metrics=("auc",),
            fold_col=fold_col,
            random_state=random_state,
        )
        scores[len(subset)] = [row["AUC"] for row in cv_result.fold_metrics]
    best = max(sorted(scores), key=lambda n: np.mean(scores[n]))
    history = []
    for subset, fitted in path(  # noqa: B007 - final fit is returned below
        train_df,
        target_col,
        features,
        model,
        model_params,
        random_state,
        step,
        min_features_to_select,
    ):
        history.append({"features": subset, "n_features": len(subset)})
        if len(subset) == best:
            break
    ranks = {f: 1 for f in subset}
    for rank, entry in enumerate(reversed(history[:-1]), 2):
        for f in entry["features"]:
            ranks.setdefault(f, rank)
    counts = sorted(scores)
    return SelectionResult(
        [
            {"feature": f, "selected": f in subset, "ranking": ranks[f]}
            for f in features
        ],
        "rfecv",
        {
            "optimal_feature_count": best,
            "model_feature_cols": subset,
            "scoring": scoring,
        },
        cv_results={
            "n_features": counts,
            "mean_test_score": [float(np.mean(scores[n])) for n in counts],
            "std_test_score": [float(np.std(scores[n])) for n in counts],
            "split_scores": [scores[n] for n in counts],
        },
        history=history,
        model=fitted.model,
    )


def sequential_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    model="logistic_regression",
    model_params=None,
    direction="forward",
    n_features_to_select=1,
    scoring="roc_auc",
    cv=5,
    fold_col=None,
    random_state=42,
):
    """Greedily add/remove features by mean native Spark validation score.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.sequential_selection(
            train_df, "target", ["income", "age"], cv=3
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/sequential_selection.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    integer(n_features_to_select, "n_features_to_select")
    if n_features_to_select >= len(features):
        raise ValueError("n_features_to_select must be smaller than feature count")
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be forward or backward")
    current = [] if direction == "forward" else features.copy()
    history = []
    if scoring != "roc_auc":
        raise ValueError(
            "sequential_selection uses Spark cross_validate metrics and "
            "requires roc_auc"
        )
    while len(current) != n_features_to_select:
        candidates = []
        for feature in features:
            if (direction == "forward") == (feature in current):
                continue
            subset = (
                [f for f in features if (f in current or f == feature)]
                if direction == "forward"
                else [f for f in current if f != feature]
            )
            cv_result = cross_validate(
                train_df,
                target_col,
                subset,
                model,
                model_params=model_params,
                cv=cv,
                metrics=("auc",),
                fold_col=fold_col,
                random_state=random_state,
            )
            candidates.append(
                {
                    "feature": feature,
                    "features": subset,
                    "mean_test_score": float(
                        np.mean([row["AUC"] for row in cv_result.fold_metrics])
                    ),
                }
            )
        winner = max(candidates, key=lambda entry: entry["mean_test_score"])
        current = winner["features"]
        history.append({**winner, "candidates": candidates})
    fitted = fit(train_df, target_col, current, model, model_params, random_state)
    return SelectionResult(
        [{"feature": f, "selected": f in current} for f in features],
        "sequential_selection",
        {"direction": direction, "model_feature_cols": current},
        history=history,
        model=fitted.model,
    )
