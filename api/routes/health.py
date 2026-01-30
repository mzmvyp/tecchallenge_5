"""
Rotas de health check para a API Passos Mágicos.
"""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends

from api.schemas import HealthResponse, ModelInfoResponse
from src.config import MODELS_DIR, settings

router = APIRouter(prefix="/health", tags=["Health"])

# Timestamp de início da aplicação
_app_start_time = datetime.now()


def get_uptime() -> float:
    """Retorna uptime em segundos."""
    return (datetime.now() - _app_start_time).total_seconds()


def check_model_exists() -> bool:
    """Verifica se o modelo existe."""
    model_path = Path(settings.model_path)
    return model_path.exists()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica o status de saúde da API"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Retorna o status de saúde da API, incluindo:
    - Status geral
    - Se o modelo está carregado
    - Versão da API
    - Tempo de atividade
    """
    model_loaded = check_model_exists()
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version=settings.model_version,
        uptime_seconds=get_uptime(),
        timestamp=datetime.now()
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness Check",
    description="Verifica se a API está pronta para receber requisições"
)
async def readiness_check() -> HealthResponse:
    """
    Readiness check para Kubernetes/Docker.
    
    Verifica se todos os componentes necessários estão prontos:
    - Modelo carregado
    - Dependências disponíveis
    """
    model_loaded = check_model_exists()
    
    # Adicionar outras verificações conforme necessário
    is_ready = model_loaded
    
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=model_loaded,
        version=settings.model_version,
        uptime_seconds=get_uptime(),
        timestamp=datetime.now()
    )


@router.get(
    "/live",
    summary="Liveness Check",
    description="Verifica se a aplicação está viva"
)
async def liveness_check() -> dict:
    """
    Liveness check para Kubernetes/Docker.
    
    Retorna um status simples indicando que a aplicação está rodando.
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Informações do Modelo",
    description="Retorna informações sobre o modelo carregado"
)
async def model_info() -> ModelInfoResponse:
    """
    Retorna informações detalhadas sobre o modelo.
    """
    model_path = Path(settings.model_path)
    
    info = ModelInfoResponse(
        model_type="RandomForest",  # Será atualizado quando modelo carregar
        version=settings.model_version,
        model_path=str(model_path),
        features=None,
        metrics=None
    )
    
    # Tentar carregar informações do modelo
    if model_path.exists():
        try:
            from src.utils.helpers import load_model
            model_data = load_model(model_path)
            
            if isinstance(model_data, dict):
                if "metadata" in model_data:
                    info.version = model_data["metadata"].get("version", info.version)
                if "cv_results" in model_data:
                    info.metrics = {
                        "best_score": model_data["cv_results"].get("best_score", 0)
                    }
        except Exception:
            pass
    
    return info
