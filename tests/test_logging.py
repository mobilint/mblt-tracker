from __future__ import annotations

import logging

from mblt_tracker._logging import suppress_pyrapl_optional_output_warnings


def test_suppresses_pyrapl_optional_output_warnings(caplog) -> None:
    logger = logging.getLogger()

    with caplog.at_level(logging.WARNING):
        with suppress_pyrapl_optional_output_warnings():
            logger.warning("imports error \n You need to install pymongo>=3.9.0 in order to use MongoOutput")
            logger.warning("imports error \n  You need to install pandas>=0.25.1 in order to use DataFrameOutput")
            logger.warning("different warning should remain visible")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["different warning should remain visible"]


def test_suppression_is_scoped(caplog) -> None:
    logger = logging.getLogger()

    with caplog.at_level(logging.WARNING):
        with suppress_pyrapl_optional_output_warnings():
            logger.warning("You need to install pymongo>=3.9.0 in order to use MongoOutput")
        logger.warning("You need to install pymongo>=3.9.0 in order to use MongoOutput")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "You need to install pymongo>=3.9.0 in order to use MongoOutput"
    ]
