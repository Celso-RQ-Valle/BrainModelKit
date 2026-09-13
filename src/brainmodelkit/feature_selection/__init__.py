"""Feature selection with optional Spark and Pandas backends.

Usage, method algorithms, parameters, and examples:
https://github.com/Celso-RQ-Valle/BrainModelKit/blob/main/docs/feature_selection.md
"""

from ._common import RFEResult
from ._selection import SelectionResult

__all__ = ["RFEResult", "SelectionResult"]
