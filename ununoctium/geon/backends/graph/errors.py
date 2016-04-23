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

class RedefiningConstantError(Error):
    """
    Redefining a constant value.
    """

class NameException(Exception):
    """
    Trying to set a value in a name generator.
    """
    pass
