"""Unit tests for Flask API health route."""


def test_health_endpoint(api_client):
    """Verify /health endpoint returns 200 OK and healthy status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "MedVision-AI API"
