"""Pytest options: ``--fast`` skips the family cells and family arms."""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--fast", action="store_true", default=False,
                     help="skip the family cells (internal) and family arms (external)")


@pytest.fixture(scope="session")
def fast(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--fast"))
