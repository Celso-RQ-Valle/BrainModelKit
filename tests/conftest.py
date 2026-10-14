"""CI requires every collected test, including optional integrations, to run."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fail-on-skip", action="store_true", help="Fail on collection or test skips"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("--fail-on-skip") and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"Required test skipped: {report.longrepr}"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    if collector.config.getoption("--fail-on-skip") and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"Required test module skipped: {report.longrepr}"
