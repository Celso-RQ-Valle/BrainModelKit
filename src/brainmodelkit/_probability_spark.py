"""Native SQL validation; no Python UDFs or SparkContext required."""


def column(name):
    from pyspark.sql import functions as F

    return F.col("`" + name.replace("`", "``") + "`")


def numeric(frame, name, policy="raise", bounds=None):
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType

    if not isinstance(frame.schema[name].dataType, NumericType):
        raise TypeError(f"{name} must be numeric")
    value = column(name).cast("double")
    valid = value.isNotNull() & ~F.isnan(value) & (F.abs(value) < float("inf"))
    if bounds is not None:
        valid = valid & value.between(*bounds)
    if policy == "raise" and frame.where(~valid).limit(1).count():
        raise ValueError(f"{name}: missing, non-finite or out-of-range values")
    return F.when(valid, value)


def sigmoid(value):
    from pyspark.sql import functions as F

    exp = F.exp(-F.abs(value))
    return F.when(value >= 0, 1 / (1 + exp)).otherwise(exp / (1 + exp))


def transformed(config, frame, name):
    from pyspark.sql import functions as F

    x = numeric(
        frame,
        name,
        config.invalid_policy,
        None if config.input_space == "raw" else (0, 1),
    )
    if config.input_space == "raw":
        return x
    if config.input_probability == "non_event":
        x = 1 - x
    if config.input_space == "logit":
        safe = F.when(
            x.isNotNull(),
            F.greatest(F.lit(config.epsilon), F.least(F.lit(1 - config.epsilon), x)),
        )
        x = F.log(safe) - F.log1p(-safe)
    return x


def labels_and_weights(frame, target, event_label, weight, policy):
    from pyspark.sql import functions as F

    label = column(target)
    missing = label.isNull()
    # Numeric NaN labels are also missing; avoid casts of string labels in ANSI SQL.
    from pyspark.sql.types import NumericType

    if isinstance(frame.schema[target].dataType, NumericType):
        missing = missing | F.isnan(label)
    if policy == "raise" and frame.where(missing).limit(1).count():
        raise ValueError("Missing targets")
    classes = [
        row[0]
        for row in frame.where(~missing).select(label).distinct().limit(3).collect()
    ]
    if len(classes) != 2 or event_label not in classes:
        raise ValueError("Binary targets must contain event_label and one other class")
    weights = F.lit(1.0) if weight is None else numeric(frame, weight)
    if frame.where(weights < 0).limit(1).count():
        raise ValueError("Weights must be non-negative")
    return F.when(~missing, (label == F.lit(event_label)).cast("double")), weights
