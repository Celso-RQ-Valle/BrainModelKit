"""Distributed repeated-fit stability and shadow-feature hypothesis testing.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

import numpy as np
from pyspark.sql import functions as F

from ._selection import SelectionResult, groups, integer, number, resolve_frame
from ._spark_model import feature_importance_selection, l1_selection
from ._spark_quality import col
from ._spark_search import indexed, owned_cache, shuffled
from ._spark_statistics import supervised


def stability_selection(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    method="feature_importance",
    n_iterations=20,
    sample_fraction=0.8,
    min_frequency=0.8,
    random_state=42,
    group_by=None,
    selector_kwargs=None,
    max_groups=100,
):
    """Measure selection frequencies across Spark Poisson bootstrap samples.

    Groups receive equal numbers of fits. See docs/feature_selection/spark.md
    #stability_selection for sampling differences and the diagnostic contract.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.stability_selection(
            train_df, "target", ["income", "age"], n_iterations=5
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/stability_selection.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    names = groups(train_df, group_by)
    integer(n_iterations, "n_iterations")
    integer(max_groups, "max_groups")
    number(sample_fraction, "sample_fraction", maximum=1)
    number(min_frequency, "min_frequency", maximum=1)
    if sample_fraction == 0:
        raise ValueError("sample_fraction must be > 0")
    if method not in ("feature_importance", "l1"):
        raise ValueError("method must be feature_importance or l1")
    options = dict(selector_kwargs or {})
    forbidden = {
        "random_state",
        "df",
        "train_df",
        "target_col",
        "feature_cols",
        "oot_df",
        "df_oot",
        "df_scoring",
        "run_name",
    }
    if options.keys() & forbidden:
        raise ValueError(
            "selector_kwargs must contain only base-selector configuration"
        )
    if method == "feature_importance":
        if not options:
            options["top_k"] = max(1, len(features) // 2)
        if options.get("top_k") is None and options.get("threshold") is None:
            raise ValueError("Specify top_k or threshold in selector_kwargs")
    keys = (
        train_df.select(*[col(n) for n in names])
        .distinct()
        .limit(max_groups + 1)
        .collect()
        if names
        else [None]
    )
    if len(keys) > max_groups:
        raise ValueError("group_by exceeds max_groups")
    counts = np.zeros(len(features), dtype=int)
    values, history = [], []
    for group_index, key in enumerate(keys):
        frame = train_df
        if key is not None:
            for name, value in zip(names, key, strict=True):
                frame = frame.filter(col(name).eqNullSafe(F.lit(value)))
        supervised(frame, target_col, features)
        for iteration in range(n_iterations):
            seed = random_state + group_index * n_iterations + iteration
            sampled = frame.sample(
                withReplacement=True, fraction=sample_fraction, seed=seed
            )
            with owned_cache(sampled) as sample:
                # Fail explicitly if a small bootstrap loses a class; do not silently
                # condition the bootstrap distribution by retrying until it passes.
                result = (
                    feature_importance_selection(
                        sample, target_col, features, random_state=seed, **options
                    )
                    if method == "feature_importance"
                    else l1_selection(sample, target_col, features, **options)
                )
                counts += np.array(
                    [r["selected"] for r in result.feature_table], dtype=int
                )
                values.append([abs(r["importance"]) for r in result.feature_table])
                history.append(
                    {
                        "group": dict(zip(names, key, strict=True))
                        if key is not None
                        else None,
                        "iteration": iteration + 1,
                        "selected_features": result.selected_features,
                    }
                )
    values = np.asarray(values)
    rows = [
        {
            "feature": f,
            "selection_count": int(counts[i]),
            "selection_frequency": float(counts[i] / len(values)),
            "mean_importance": float(values[:, i].mean()),
            "std_importance": float(values[:, i].std()),
            "selected": bool(counts[i] / len(values) >= min_frequency),
        }
        for i, f in enumerate(features)
    ]
    return SelectionResult(
        rows,
        "stability_selection",
        {
            "n_fits": len(values),
            "method": method,
            "min_frequency": min_frequency,
            "sampling": "poisson_bootstrap",
            "random_state": random_state,
        },
        history=history,
    )


def decisions(hits, iteration, alpha, two_step, total_features):
    """Return binomial tails and corrected acceptance/rejection masks."""
    from scipy.stats import binom

    accept_p = binom.sf(np.asarray(hits) - 1, iteration, 0.5)
    reject_p = binom.cdf(hits, iteration, 0.5)

    def corrected(p):
        if not two_step:
            return p <= alpha / total_features
        order = np.argsort(p, kind="stable")
        passing = p[order] <= alpha * np.arange(1, len(p) + 1) / len(p)
        keep = np.zeros(len(p), dtype=bool)
        if passing.any():
            keep[order[: np.flatnonzero(passing)[-1] + 1]] = True
        return keep & (p <= alpha / iteration)

    return accept_p, reject_p, corrected(accept_p), corrected(reject_p)


def boruta(
    train_df=None,
    target_col=None,
    feature_cols=None,
    *,
    df=None,
    model="random_forest",
    model_params=None,
    n_estimators=100,
    max_iter=100,
    perc=100,
    alpha=0.05,
    two_step=True,
    random_state=42,
):
    """Test real importance against shuffled shadows using binomial hit tests.

    Only confirmed features are selected. Spark uses native random forests and
    distributed permutations. See docs/feature_selection/boruta.md for the exact
    hypotheses, corrections, diagnostics, examples, and backend differences.

    Usage
    -----
    Import `brainmodelkit.feature_selection.pyspark` as `fs`, then call:

        result = fs.boruta(
            train_df, "target", ["income", "age"], n_estimators=30, max_iter=30
        )
        print(result.selected_features)

    Parameters, return fields, algorithm, assumptions, and complete examples:
    https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection/boruta.md
    """
    train_df = resolve_frame(train_df, df, "train_df", "df")
    features = supervised(train_df, target_col, feature_cols)
    integer(max_iter, "max_iter")
    integer(n_estimators, "n_estimators")
    number(perc, "perc", maximum=100)
    number(alpha, "alpha", maximum=1)
    if perc == 0 or alpha == 0 or alpha == 1:
        raise ValueError("perc must be > 0 and alpha must be strictly between 0 and 1")
    if type(two_step) is not bool:
        raise ValueError("two_step must be Boolean")
    if not isinstance(model, str) or model != "random_forest":
        raise ValueError("Spark Boruta currently supports model='random_forest'")
    if any(f.startswith("__bmk_") for f in [*features, target_col]):
        raise ValueError("Names beginning __bmk_ are reserved for shadows")
    params = dict(model_params or {})
    if "numTrees" in params:
        raise ValueError("Use n_estimators instead of model_params['numTrees']")
    params["numTrees"] = n_estimators
    status = np.zeros(len(features), dtype=int)
    hits = np.zeros(len(features), dtype=int)
    p_accept = np.ones(len(features))
    p_reject = np.ones(len(features))
    trials = np.zeros(len(features), dtype=int)
    history = []
    with indexed(train_df.select(*[col(f) for f in [*features, target_col]])) as base:
        for iteration in range(1, max_iter + 1):
            active = np.flatnonzero(status >= 0)
            active_names = [features[i] for i in active]
            augmented = base.select(
                "__bmk_id", *[col(f) for f in [*active_names, target_col]]
            )
            shadow_names = []
            for j in range(max(5, len(active))):
                name = f"__bmk_shadow_{j}"
                # Each independent permutation starts from the same stable row IDs.
                shadow = shuffled(
                    base,
                    active_names[j % len(active)],
                    random_state + iteration * max(5, len(features)) + j,
                    name,
                ).select("__bmk_id", name)
                augmented = augmented.join(shadow, "__bmk_id")
                shadow_names.append(name)
            with owned_cache(augmented) as cached:
                fitted = feature_importance_selection(
                    cached,
                    target_col,
                    [*active_names, *shadow_names],
                    model=model,
                    model_params=params,
                    random_state=random_state + iteration,
                )
                importance = {
                    r["feature"]: r["importance"] for r in fitted.feature_table
                }
                shadow_threshold = float(
                    np.percentile([importance[f] for f in shadow_names], perc)
                )
                hits[active] += np.array(
                    [importance[f] > shadow_threshold for f in active_names], dtype=int
                )
                trials[active] = iteration
                pa, pr, accept, reject = decisions(
                    hits[active], iteration, alpha, two_step, len(features)
                )
                p_accept[active], p_reject[active] = pa, pr
                for j, i in enumerate(active):
                    if status[i] == 0:
                        if accept[j]:
                            status[i] = 1
                        elif reject[j]:
                            status[i] = -1
                history.append(
                    {
                        "iteration": iteration,
                        "shadow_threshold": shadow_threshold,
                        "importance": {f: importance[f] for f in active_names},
                        "hits": dict(zip(features, hits.tolist(), strict=True)),
                    }
                )
            if not (status == 0).any():
                break
    rows = [
        {
            "feature": f,
            "status": {0: "tentative", 1: "confirmed", -1: "rejected"}[int(status[i])],
            "selected": bool(status[i] == 1),
            "ranking": {1: 1, 0: 2, -1: 3}[int(status[i])],
            "hits": int(hits[i]),
            "n_trials": int(trials[i]),
            "p_accept": float(p_accept[i]),
            "p_reject": float(p_reject[i]),
        }
        for i, f in enumerate(features)
    ]
    return SelectionResult(
        rows,
        "boruta",
        {
            "n_iterations": len(history),
            "perc": perc,
            "alpha": alpha,
            "two_step": two_step,
            "random_state": random_state,
            "ranking_semantics": "status_groups",
            "rough_fix": False,
        },
        history=history,
    )
