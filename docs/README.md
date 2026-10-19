# BrainModelKit documentation

Start with the [Spark journey](../README.md#spark-guide) or
[pandas journey](../README.md#pandas-guide). Both move from installation to model
lifecycle; their parameters and algorithms are not interchangeable.

| Stage | Practical guide | Detailed reference |
| --- | --- | --- |
| Install and diagnose | [Setup](setup.md) | [Compatibility and release checks](release-readiness.md) |
| Prepare data | [Runnable walkthroughs](../examples/journey/README.md) | [Standalone simulations](standalone-examples.md) |
| Build features | [Temporal catalog](temporal.md) | [Boundaries and scale](temporal.md#temporal-semantics) |
| Select features | [Cost and purpose comparison](selection-costs.md) | [All sixteen method guides](feature_selection.md#api-quick-reference) |
| Train | [Backend examples](../examples/journey/README.md) | [Models, parameters, outputs](training.md) |
| Validate and optimize | [Leakage-aware alternatives](journey-validation.md) | [CV](model_selection.md), [Optuna](optimization.md) |
| Evaluate | [Metric examples](evaluation.md) | [Aliases, scalar KS, undefined cases](evaluation.md#public-alias-and-scalar-utility) |
| Analyze segments | [Cube interpretation](evaluation.md#interpreting-segment-results) | [Cube schemas](evaluation.md#output-conventions) |
| Save, load, score | [Runnable lifecycle](../examples/journey/README.md) | [Formats](training.md#model-formats), [loaders](training.md#loading-trained-models), [MLflow](training.md#mlflow) |
| Troubleshoot | [Environment checks](setup.md#environment-doctor) | [Windows](setup.md#spark-on-windows), [Spark limits](setup.md#spark-scalability-limits), [migration](training.md#migration-and-compatibility) |
| Contribute | [Contributing](../CONTRIBUTING.md) | [Project reference](project.md) |

For previous README topics, use the [content move map](navigation.md).
