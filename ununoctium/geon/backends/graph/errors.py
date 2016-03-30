class Error(Exception):
    """
    Base class for graph errors.
    """
    pass


class MissingGraphError(Error):
    """
    Graph cannot be determined.
    """

class UnititializedVariableError(Error):
    """
    Attempt to use the value of an unitialized variable.
    """

class IncompatibleShapesError(Error):
    """
    Incompatible shapes.
    """

class IncompatibleTypesError(Error):
    """
    Incompatible graph types.
    """
