# LightGBM: parameter reference

[Model examples](README.md#lightgbm) | [Search-space syntax](../optimization.md#search-spaces) | [Training and refitting](../training.md)

Omit a key to keep its default; an empty list or an empty string does not request
the default. `[value]` fixes a scalar, and `[minimum, maximum]` tunes a numeric
interval. `None` is an explicit value, not an instruction to use the default.
The catalog below is commented out so that optional settings are not passed as
explicit nulls. Uncomment only the entries you want to adjust, and supply at
least one entry to `optimize`. For `train_model(model_params=...)`, use scalar
values directly, without the search-space brackets.

These catalogs include all explicit constructor parameters in the verified local
versions, except Spark columns managed by BrainModelKit and the internal Java
handle. Some parameters control execution or prediction rather than learning.
Defaults may change with dependency versions; the linked API explains each
parameter, its valid values, and compatibility constraints. Custom estimators
retain their configured values when an entry is omitted. Built-in model strings
use the optimization seed (42 by default) for `random_state` / `seed`.

## pandas

[Parameter meanings and valid combinations](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html). Catalog: 4.6.0.

```python
search_space = {
    # "boosting_type": ['gbdt'],  # Default if omitted: 'gbdt'
    # "num_leaves": [31],  # Default if omitted: 31
    # "max_depth": [-1],  # Default if omitted: -1
    # "learning_rate": [0.1],  # Default if omitted: 0.1
    # "n_estimators": [100],  # Default if omitted: 100
    # "subsample_for_bin": [200000],  # Default if omitted: 200000
    # "objective": [None],  # Default if omitted: None
    # "class_weight": [None],  # Default if omitted: None
    # "min_split_gain": [0.0],  # Default if omitted: 0.0
    # "min_child_weight": [0.001],  # Default if omitted: 0.001
    # "min_child_samples": [20],  # Default if omitted: 20
    # "subsample": [1.0],  # Default if omitted: 1.0
    # "subsample_freq": [0],  # Default if omitted: 0
    # "colsample_bytree": [1.0],  # Default if omitted: 1.0
    # "reg_alpha": [0.0],  # Default if omitted: 0.0
    # "reg_lambda": [0.0],  # Default if omitted: 0.0
    # "random_state": [None],  # Default if omitted: None; built-in model string: optimization seed (default 42)
    # "n_jobs": [None],  # Default if omitted: None
    # "importance_type": ['split'],  # Default if omitted: 'split'
}
```

## Spark / SynapseML

[Parameter meanings and valid combinations](https://mmlspark.blob.core.windows.net/docs/1.0.14/pyspark/synapse.ml.lightgbm.html). Catalog: 2.139.0.

```python
search_space = {
    # "baggingFraction": [1.0],  # Default if omitted: 1.0
    # "baggingFreq": [0],  # Default if omitted: 0
    # "baggingSeed": [3],  # Default if omitted: 3
    # "binSampleCount": [200000],  # Default if omitted: 200000
    # "boostFromAverage": [True],  # Default if omitted: True
    # "boostingType": ['gbdt'],  # Default if omitted: 'gbdt'
    # "catSmooth": [10.0],  # Default if omitted: 10.0
    # "categoricalSlotIndexes": lambda trial: [],  # Default if omitted: []; structured value
    # "categoricalSlotNames": lambda trial: [],  # Default if omitted: []; structured value
    # "catl2": [10.0],  # Default if omitted: 10.0
    # "chunkSize": [10000],  # Default if omitted: 10000
    # "dataRandomSeed": [1],  # Default if omitted: 1
    # "dataTransferMode": ['streaming'],  # Default if omitted: 'streaming'
    # "defaultListenPort": [12400],  # Default if omitted: 12400
    # "deterministic": [False],  # Default if omitted: False
    # "driverListenPort": [0],  # Default if omitted: 0
    # "dropRate": [0.1],  # Default if omitted: 0.1
    # "dropSeed": [4],  # Default if omitted: 4
    # "earlyStoppingRound": [0],  # Default if omitted: 0
    # "executionMode": [None],  # Default if omitted: None
    # "extraSeed": [6],  # Default if omitted: 6
    # "featureFraction": [1.0],  # Default if omitted: 1.0
    # "featureFractionByNode": [None],  # Default if omitted: None
    # "featureFractionSeed": [2],  # Default if omitted: 2
    # "featuresShapCol": [''],  # Default if omitted: ''
    # "fobj": [None],  # Default if omitted: None
    # "improvementTolerance": [0.0],  # Default if omitted: 0.0
    # "initScoreCol": [None],  # Default if omitted: None
    # "isEnableSparse": [True],  # Default if omitted: True
    # "isProvideTrainingMetric": [False],  # Default if omitted: False
    # "isUnbalance": [False],  # Default if omitted: False
    # "lambdaL1": [0.0],  # Default if omitted: 0.0
    # "lambdaL2": [0.0],  # Default if omitted: 0.0
    # "leafPredictionCol": [''],  # Default if omitted: ''
    # "learningRate": [0.1],  # Default if omitted: 0.1
    # "matrixType": ['auto'],  # Default if omitted: 'auto'
    # "maxBin": [255],  # Default if omitted: 255
    # "maxBinByFeature": lambda trial: [],  # Default if omitted: []; structured value
    # "maxCatThreshold": [32],  # Default if omitted: 32
    # "maxCatToOnehot": [4],  # Default if omitted: 4
    # "maxDeltaStep": [0.0],  # Default if omitted: 0.0
    # "maxDepth": [-1],  # Default if omitted: -1
    # "maxDrop": [50],  # Default if omitted: 50
    # "maxNumClasses": [100],  # Default if omitted: 100
    # "maxStreamingOMPThreads": [16],  # Default if omitted: 16
    # "metric": [''],  # Default if omitted: ''
    # "microBatchSize": [100],  # Default if omitted: 100
    # "minDataInLeaf": [20],  # Default if omitted: 20
    # "minDataPerBin": [3],  # Default if omitted: 3
    # "minDataPerGroup": [100],  # Default if omitted: 100
    # "minGainToSplit": [0.0],  # Default if omitted: 0.0
    # "minSumHessianInLeaf": [0.001],  # Default if omitted: 0.001
    # "modelString": [''],  # Default if omitted: ''
    # "monotoneConstraints": lambda trial: [],  # Default if omitted: []; structured value
    # "monotoneConstraintsMethod": ['basic'],  # Default if omitted: 'basic'
    # "monotonePenalty": [0.0],  # Default if omitted: 0.0
    # "negBaggingFraction": [1.0],  # Default if omitted: 1.0
    # "numBatches": [0],  # Default if omitted: 0
    # "numIterations": [100],  # Default if omitted: 100
    # "numLeaves": [31],  # Default if omitted: 31
    # "numTasks": [0],  # Default if omitted: 0
    # "numThreads": [0],  # Default if omitted: 0
    # "objective": ['binary'],  # Default if omitted: 'binary'
    # "objectiveSeed": [5],  # Default if omitted: 5
    # "otherRate": [0.1],  # Default if omitted: 0.1
    # "parallelism": ['data_parallel'],  # Default if omitted: 'data_parallel'
    # "passThroughArgs": [''],  # Default if omitted: ''
    # "posBaggingFraction": [1.0],  # Default if omitted: 1.0
    # "predictDisableShapeCheck": [False],  # Default if omitted: False
    # "referenceDataset": [None],  # Default if omitted: None
    # "repartitionByGroupingColumn": [True],  # Default if omitted: True
    # "samplingMode": ['subset'],  # Default if omitted: 'subset'
    # "samplingSubsetSize": [1000000],  # Default if omitted: 1000000
    # "seed": [None],  # Default if omitted: None; built-in model string: optimization seed (default 42)
    # "skipDrop": [0.5],  # Default if omitted: 0.5
    # "slotNames": lambda trial: [],  # Default if omitted: []; structured value
    # "thresholds": [None],  # Default if omitted: None
    # "timeout": [1200.0],  # Default if omitted: 1200.0
    # "topK": [20],  # Default if omitted: 20
    # "topRate": [0.2],  # Default if omitted: 0.2
    # "uniformDrop": [False],  # Default if omitted: False
    # "useBarrierExecutionMode": [False],  # Default if omitted: False
    # "useMissing": [True],  # Default if omitted: True
    # "useSingleDatasetMode": [True],  # Default if omitted: True
    # "validationIndicatorCol": [None],  # Default if omitted: None
    # "verbosity": [-1],  # Default if omitted: -1
    # "weightCol": [None],  # Default if omitted: None
    # "xGBoostDartMode": [False],  # Default if omitted: False
    # "zeroAsMissing": [False],  # Default if omitted: False
}
```

## SparkLightGBM (default Spark backend)

This backend accepts native LightGBM parameters rather than SynapseML names.
[All native parameters, defaults, aliases, and meanings](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
form the complete reference, including objective-specific and device-specific
options accepted through keyword arguments. Avoid specifying multiple aliases
for the same setting. [Backend setup](../optimization.md#spark-lightgbm) explains
worker allocation and dependencies.

```python
search_space = {
    "num_iterations": [100, 200],  # Default if omitted: 100
    "learning_rate": [0.03, 0.1],  # Default if omitted: 0.1
    "num_leaves": [15, 63],  # Default if omitted: 31
    "max_depth": [-1],  # Default if omitted: -1 (unlimited)
    "min_data_in_leaf": [200, 1000],  # Default if omitted: 20
    "min_sum_hessian_in_leaf": [0.001],  # Default if omitted: 0.001
    "min_gain_to_split": [0.0],  # Default if omitted: 0.0
    "lambda_l1": [0.0],  # Default if omitted: 0.0
    "lambda_l2": [0.0],  # Default if omitted: 0.0
    "feature_fraction": [1.0],  # Default if omitted: 1.0
    "bagging_fraction": [1.0],  # Default if omitted: 1.0
    "bagging_freq": [0],  # Default if omitted: 0 (disabled)
    "max_bin": [255],  # Default if omitted: 255
    "is_unbalance": [False],  # Default if omitted: False
    "scale_pos_weight": [1.0],  # Default if omitted: 1.0
    "seed": [42],  # Default if omitted: optimization seed (42)
}
```

Enable bagging with both `bagging_freq > 0` and `bagging_fraction < 1`.
Do not enable `is_unbalance` and change `scale_pos_weight` together. Early
stopping requires a validation dataset supported by the estimator's fit API;
the optimization `oot_df` is used for evaluation, not automatically for early
stopping. The same caveat applies to SynapseML validation columns.
