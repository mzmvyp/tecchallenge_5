"""Rotas da API."""

from .predict import router as predict_router
from .health import router as health_router
from .monitoring import router as monitoring_router

__all__ = ["predict_router", "health_router", "monitoring_router"]
