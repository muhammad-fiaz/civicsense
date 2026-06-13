"""FastAPI application for CivicSense.

Provides REST API endpoints for image/video detection,
camera management, incidents, and analytics.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from civicsense.api.routes import router
from civicsense.core.logging import get_logger

logger = get_logger("app")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="CivicSense API",
        description="AI-powered littering detection API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unhandled exceptions."""
        logger.error(f"Unhandled API error: {exc}", module="app")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": str(request.url),
            },
        )

    app.include_router(router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "civicsense"}

    return app
