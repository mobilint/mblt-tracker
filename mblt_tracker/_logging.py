from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

_PYRAPL_OPTIONAL_OUTPUT_WARNING_FRAGMENTS = (
    "You need to install pymongo>=3.9.0 in order to use MongoOutput",
    "You need to install pandas>=0.25.1 in order to use DataFrameOutput",
)


class _PyRAPLOptionalOutputWarningFilter(logging.Filter):
    """Filter pyRAPL optional output backend warnings unused by mblt-tracker."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            fragment in message
            for fragment in _PYRAPL_OPTIONAL_OUTPUT_WARNING_FRAGMENTS
        )


@contextmanager
def suppress_pyrapl_optional_output_warnings() -> Iterator[None]:
    """Suppress pyRAPL Mongo/DataFrame optional backend warnings only.

    pyRAPL logs warnings while importing optional output backends when pymongo or
    pandas are not installed. mblt-tracker does not use those output backends, so
    those warnings are noise. Keep the suppression scoped and message-specific so
    unrelated logging from pyRAPL or other packages remains visible.
    """

    root_logger = logging.getLogger()
    log_filter = _PyRAPLOptionalOutputWarningFilter()
    root_logger.addFilter(log_filter)
    try:
        yield
    finally:
        root_logger.removeFilter(log_filter)
