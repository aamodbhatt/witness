"""Exception hierarchy. Every error raised on purpose carries an actionable message."""


class WitnessError(Exception):
    """Base class for all errors raised deliberately by witness."""


class ConfigError(WitnessError):
    """The configuration is missing, malformed or self-contradictory."""


class IngestError(WitnessError):
    """A document could not be loaded or chunked."""


class IndexStoreError(WitnessError):
    """The on-disk index is missing, corrupt or incompatible with the current config."""


class BackendError(WitnessError):
    """A model backend failed (bad response, unreachable endpoint, unsupported model)."""
