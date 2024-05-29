"""A set of customized exceptions for the ML framework."""


class DimensionError(Exception):
    """Handling dimension mismatch errors."""

    def __init__(self, input_dimension, required_dimension):
    	"""Compares given dimension with expected ones."""
        message = f"Dimension mismatch. Expected shape ({required_dimension})"
        message += f", but got instead shape ({input_dimension})."
        self.message = message
        super().__init__(self.message)


class SnippetsKeyMismatch(Exception):
	"""The keys of one snippet do not match the other.""" 


class MissingRequiredFeatures(Exception):
	"""input dict is missing required feature names."""

	def __init__(self, provided_features: set, required_features: set):
		"""Compares provided features with expected ones."""
		missing_keys = required_features - provided_features
		message = f"You are missing one or more required features: "
		message += f"{missing_keys}"
		self.message = message
		super().__init__(self.message)

