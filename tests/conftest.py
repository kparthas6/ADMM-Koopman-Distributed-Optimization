import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("experiments", "controller", "common"):
    sys.path.insert(0, os.path.join(ROOT, d))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="also run the closed-loop MuJoCo tests (minutes)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: closed-loop MuJoCo runs; enabled with --runslow")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip = pytest.mark.skip(reason="closed-loop MuJoCo test: use --runslow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
