class Ds4Error(RuntimeError):
    """Base error for pyds4 failures."""


class Ds4ApiVersionError(Ds4Error):
    """Raised when the DS4 native API does not match the binding."""


class Ds4BackendUnavailable(Ds4Error):
    """Raised when a requested native backend cannot be used."""


class Ds4LoadError(Ds4Error):
    """Raised when a DS4 engine cannot load a model."""


class Ds4InvalidModel(Ds4Error):
    """Raised when a DS4 model path or file is invalid."""


class Ds4ContextError(Ds4Error):
    """Raised when a DS4 context size or session state is invalid."""


class Ds4GenerationError(Ds4Error):
    """Raised when DS4 generation fails."""


class Ds4Cancelled(Ds4Error):
    """Raised when DS4 generation is cancelled."""
