"""FastAPI application factory and server entrypoint for MedVision-AI."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from medvision.api.routes import router as api_router
from medvision.api.services import ModelService
from medvision.utils.logger import get_logger

logger = get_logger("medvision.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load model on startup and clean up on shutdown."""
    logger.info("Initializing MedVision-AI API server...")
    service = ModelService.get_instance()
    try:
        service.initialize()
        logger.info(f"Model initialized: loaded={service.is_loaded()} on {service.device}")
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
    yield
    logger.info("Shutting down MedVision-AI API server.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="🩺 MedVision-AI REST API",
        description=(
            "Enterprise-grade REST API for DenseNet121 chest radiograph pneumonia detection "
            "and Grad-CAM visual explainability. Research and educational prototype only."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Router
    app.include_router(api_router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    return app


app = create_app()


def main():
    """Run uvicorn server directly."""
    import uvicorn
    uvicorn.run("medvision.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
