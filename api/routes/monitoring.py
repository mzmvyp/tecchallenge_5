"""
Rotas de monitoramento para a API Passos Mágicos.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from api.schemas import DriftReportResponse, ErrorResponse, MetricsResponse
from src.monitoring.metrics_tracker import get_metrics_tracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Métricas do Sistema",
    description="Retorna métricas de performance e uso do sistema"
)
async def get_metrics() -> MetricsResponse:
    """
    Retorna métricas agregadas do sistema:
    - Total de predições
    - Taxa de erros
    - Distribuição de predições por classe
    - Estatísticas de latência
    """
    tracker = get_metrics_tracker()
    summary = tracker.get_metrics_summary()
    
    return MetricsResponse(
        total_predictions=summary["total_predictions"],
        total_errors=summary["total_errors"],
        error_rate=summary["error_rate"],
        prediction_distribution=summary["prediction_distribution"],
        latency=summary.get("latency"),
        predictions_per_minute=summary["predictions_per_minute"],
        session_uptime_seconds=summary["session_uptime_seconds"]
    )


@router.get(
    "/metrics/daily",
    summary="Métricas Diárias",
    description="Retorna métricas de um dia específico"
)
async def get_daily_metrics(
    date: Optional[str] = Query(
        None,
        description="Data no formato YYYY-MM-DD (padrão: hoje)",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
) -> dict:
    """
    Retorna métricas de predições para um dia específico.
    """
    tracker = get_metrics_tracker()
    return tracker.get_daily_metrics(date)


@router.get(
    "/metrics/latency",
    summary="Histograma de Latência",
    description="Retorna distribuição de latências das predições"
)
async def get_latency_histogram(
    bins: int = Query(default=20, ge=5, le=100, description="Número de bins")
) -> dict:
    """
    Retorna histograma de latências das predições.
    """
    tracker = get_metrics_tracker()
    return tracker.get_latency_histogram(bins)


@router.get(
    "/errors",
    summary="Erros Recentes",
    description="Retorna lista de erros recentes"
)
async def get_recent_errors(
    n: int = Query(default=10, ge=1, le=100, description="Número de erros")
) -> dict:
    """
    Retorna os N erros mais recentes registrados.
    """
    tracker = get_metrics_tracker()
    errors = tracker.get_recent_errors(n)
    
    return {
        "total": len(errors),
        "errors": errors
    }


@router.get(
    "/drift-report",
    response_model=DriftReportResponse,
    summary="Relatório de Drift",
    description="Retorna informações sobre drift detectado"
)
async def get_drift_report() -> DriftReportResponse:
    """
    Retorna o relatório mais recente de drift detectado.
    
    Para gerar um novo relatório de drift, use POST /monitoring/drift-check.
    """
    try:
        from src.monitoring.drift_detector import DriftDetector
        
        detector = DriftDetector()
        summary = detector.get_drift_summary()
        
        # Retornar informações do último relatório
        return DriftReportResponse(
            drift_detected=False,  # Será atualizado quando executar verificação
            features_with_drift=[],
            overall_drift_score=0.0,
            report_path=summary.get("reports_dir"),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Erro ao obter relatório de drift: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter relatório: {str(e)}"
        )


@router.post(
    "/drift-check",
    summary="Verificar Drift",
    description="Executa verificação de drift nos dados recentes"
)
async def check_drift() -> dict:
    """
    Executa verificação de drift comparando dados recentes
    com os dados de referência (treino).
    
    Requer que o modelo tenha sido treinado previamente.
    """
    return {
        "status": "not_implemented",
        "message": "Endpoint em desenvolvimento. "
                   "Use o módulo DriftDetector diretamente para análises de drift."
    }


@router.post(
    "/metrics/reset",
    summary="Resetar Métricas",
    description="Reseta todas as métricas coletadas"
)
async def reset_metrics() -> dict:
    """
    Reseta todas as métricas da sessão atual.
    
    **Atenção**: Esta ação não pode ser desfeita.
    """
    tracker = get_metrics_tracker()
    tracker.reset()
    
    return {
        "status": "success",
        "message": "Métricas resetadas com sucesso",
        "timestamp": datetime.now().isoformat()
    }


@router.post(
    "/metrics/save",
    summary="Salvar Métricas",
    description="Persiste métricas em disco"
)
async def save_metrics() -> dict:
    """
    Salva métricas atuais em disco para análise posterior.
    """
    tracker = get_metrics_tracker()
    filepath = tracker.save_metrics()
    
    return {
        "status": "success",
        "filepath": str(filepath) if filepath else None,
        "timestamp": datetime.now().isoformat()
    }
