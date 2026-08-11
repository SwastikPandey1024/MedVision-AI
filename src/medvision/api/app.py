"""Flask application factory and main entrypoint (Phase 9)."""

from flask import Flask
from medvision.api.routes import api_bp
from medvision.utils.logger import get_logger

logger = get_logger("medvision.api")


def create_app() -> Flask:
    """Create and configure Flask application instance."""
    app = Flask(__name__)
    app.register_blueprint(api_bp)

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Resource not found"}, 404

    @app.errorhandler(500)
    def internal_error(e):
        return {"error": "Internal server error"}, 500

    return app


def main() -> None:
    """Run local development API server."""
    app = create_app()
    logger.info("Starting MedVision-AI REST API server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
