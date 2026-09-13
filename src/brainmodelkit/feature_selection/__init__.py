"""Feature selection with optional Spark and Pandas backends.

Start with the [Feature Selection Guide](../../docs/feature_selection.md), then
import selectors from :mod:`brainmodelkit.feature_selection.pandas` or
:mod:`brainmodelkit.feature_selection.pyspark`.
"""

from ._common import RFEResult
from ._selection import SelectionResult

__all__ = ["RFEResult", "SelectionResult"]
