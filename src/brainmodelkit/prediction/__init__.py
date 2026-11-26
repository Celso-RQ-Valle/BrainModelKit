"""Composition of fitted binary model predictions."""

from copy import deepcopy


class Workflow:
    """Common metadata for prediction compositions."""

    def _inherit_metadata(self, source):
        self.feature_space = getattr(source, "feature_space", "original")
        self.feature_dictionary = deepcopy(getattr(source, "feature_dictionary", {}))
        self.background_reference = deepcopy(
            getattr(source, "background_reference", None)
        )
