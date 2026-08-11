"""Pytest shared fixtures and setup configuration."""

import pytest
from medvision.config.settings import load_config


@pytest.fixture
def sample_config():
    """Fixture providing loaded project configuration dict."""
    return load_config()


@pytest.fixture
def api_client():
    """Fixture providing Flask test client instance."""
    try:
        from medvision.api.app import create_app
    except ImportError:
        pytest.skip("Flask is not installed in the environment yet.")
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

