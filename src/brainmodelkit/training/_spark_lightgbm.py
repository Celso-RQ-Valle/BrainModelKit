"""Spark ML adapter for SparkLightGBM 0.9.2's public native-model API.

The native model and its sidecar are stored as Spark Params so PipelineModel's
Python-stage writer/reader can round-trip them without pickling native objects.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pyspark.ml import Estimator, Model
from pyspark.ml.functions import array_to_vector
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


def require_sparklightgbm():
    try:
        import sparklightgbm
    except ImportError as exc:
        raise ImportError(
            "lgbm_backend='default' requires sparklightgbm>=0.9.2,<0.10. "
            "Install 'brainmodelkit[sparklightgbm]' on the driver and every "
            "executor. Use lgbm_backend='synapseml' explicitly for SynapseML."
        ) from exc
    from brainmodelkit.environment.capabilities import READY, version_status

    version = getattr(sparklightgbm, "__version__", None)
    if (
        version_status(version, "0.9.2") != READY
        or version_status(version, "0.10") == READY
    ):
        raise ImportError(
            f"Unsupported SparkLightGBM version {version!r}; install "
            "'sparklightgbm>=0.9.2,<0.10' on the driver and executors."
        )
    return sparklightgbm


class SparkLightGBMEstimator(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    options = Param(Params._dummy(), "options", "SparkLightGBM constructor arguments")

    def __init__(self, options=None):
        super().__init__()
        self._set(options=dict(options or {}))

    def _fit(self, dataset):
        if getattr(dataset.sparkSession, "_jvm", None) is None:
            raise RuntimeError(
                "SparkLightGBM requires classic Spark with RDD/barrier access; "
                "Spark Connect is unsupported"
            )
        fitted = (
            require_sparklightgbm()
            .LightGBMClassifier(**self.getOrDefault(self.options))
            .fit(dataset)
        )
        with TemporaryDirectory(prefix="bmk-lgbm-") as directory:
            path = Path(directory) / "model.txt"
            fitted.save_native_model(str(path))
            return SparkLightGBMModel(
                nativeModel=path.read_bytes().decode("utf-8"),
                nativeMetadata=json.loads(
                    path.with_suffix(".txt.json").read_text(encoding="utf-8")
                ),
                options=self.getOrDefault(self.options),
            )


class SparkLightGBMModel(Model, DefaultParamsReadable, DefaultParamsWritable):
    nativeModel = Param(
        Params._dummy(), "nativeModel", "Native LightGBM text", TypeConverters.toString
    )
    nativeMetadata = Param(Params._dummy(), "nativeMetadata", "Native sidecar")
    options = Param(Params._dummy(), "options", "Bridge options")

    def __init__(self, nativeModel="", nativeMetadata=None, options=None):
        super().__init__()
        self._set(
            nativeModel=nativeModel,
            nativeMetadata=nativeMetadata or {},
            options=options or {},
        )
        self._native = None

    def _restore(self):
        if self._native is None:
            with TemporaryDirectory(prefix="bmk-lgbm-") as directory:
                path = Path(directory) / "model.txt"
                # Native tree offsets are byte-based: newline translation on
                # Windows would corrupt the model, so preserve the exact bytes.
                path.write_bytes(self.getOrDefault(self.nativeModel).encode("utf-8"))
                path.with_suffix(".txt.json").write_text(
                    json.dumps(self.getOrDefault(self.nativeMetadata)), encoding="utf-8"
                )
                options = self.getOrDefault(self.options)
                overrides = {
                    k: v
                    for k, v in options.items()
                    if k
                    in {
                        "features_col",
                        "prediction_col",
                        "probability_col",
                        "raw_prediction_col",
                        "leaf_prediction_col",
                        "prediction_batch_size",
                        "num_threads",
                        "num_thread",
                        "nthread",
                        "nthreads",
                        "n_jobs",
                    }
                }
                model_class = require_sparklightgbm().LightGBMClassificationModel
                self._native = model_class.load_native_model(str(path), **overrides)
        return self._native

    def _transform(self, dataset):
        scored = self._restore().transform(dataset)
        # Bridge outputs are array<double>; the rest of BrainModelKit uses vectors.
        for name in ("probability", "rawPrediction"):
            scored = scored.withColumn(name, array_to_vector(name))
        return scored

    def getFeatureImportances(self):
        return self._restore().feature_importance("split")
