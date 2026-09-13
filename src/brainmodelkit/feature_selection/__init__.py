"""Feature selection with optional Spark and Pandas backends."""

from ._common import RFEResult
from ._selection import SelectionResult

__all__ = ["RFEResult", "SelectionResult"]
