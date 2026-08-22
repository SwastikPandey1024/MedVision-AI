"""Root API package entrypoint re-exporting MedVision-AI FastAPI application."""

from medvision.api.main import app, create_app, main
from medvision.api import schemas, routes, services

__all__ = ["app", "create_app", "main", "schemas", "routes", "services"]

if __name__ == "__main__":
    main()
