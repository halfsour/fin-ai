"""Shared pytest configuration and fixtures."""

def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run end-to-end tests against real Bedrock (requires AWS credentials)",
    )
