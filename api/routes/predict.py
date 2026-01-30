"""
Rotas de predição para a API Passos Mágicos.
"""

import time
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, status

from api.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    ErrorResponse,
    PredictionOutput,
    StudentInput,
)
from src.config import RISCO_LABELS
from src.monitoring.metrics_tracker import get_metrics_tracker, LatencyTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Predictor será carregado na inicialização
_predictor = None


def get_predictor():
    """Obtém o predictor, carregando se necessário."""
    global _predictor
    
    if _predictor is None:
        try:
            from src.models.predict import RiskPredictor
            _predictor = RiskPredictor()
            logger.info("Predictor carregado com sucesso")
        except FileNotFoundError:
            logger.warning("Modelo não encontrado. Treine o modelo primeiro.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modelo não disponível. Execute o treinamento primeiro."
            )
        except Exception as e:
            logger.error(f"Erro ao carregar predictor: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro ao carregar modelo: {str(e)}"
            )
    
    return _predictor


@router.post(
    "",
    response_model=PredictionOutput,
    summary="Predição Individual",
    description="Realiza predição de risco de defasagem para um estudante",
    responses={
        200: {"description": "Predição realizada com sucesso"},
        400: {"model": ErrorResponse, "description": "Dados inválidos"},
        503: {"model": ErrorResponse, "description": "Modelo não disponível"},
    }
)
async def predict_defasagem(student: StudentInput) -> PredictionOutput:
    """
    Realiza predição de risco de defasagem escolar para um estudante.
    
    O modelo analisa os indicadores educacionais e retorna:
    - Nível de risco (BAIXO, MÉDIO, ALTO)
    - Probabilidade da classificação
    - Probabilidades para cada classe
    """
    metrics = get_metrics_tracker()
    
    with LatencyTracker(metrics, "single_prediction") as tracker:
        try:
            predictor = get_predictor()
            
            # Realizar predição
            result = predictor.predict_risk(
                fase=student.fase,
                idade=student.idade,
                genero=student.genero,
                anos_na_pm=student.anos_na_pm,
                inde=student.inde,
                ian=student.ian,
                ida=student.ida,
                ieg=student.ieg,
                iaa=student.iaa,
                ips=student.ips,
                ipp=student.ipp,
                ipv=student.ipv,
                pedra=student.pedra,
                instituicao_ensino=student.instituicao_ensino,
                bolsista=student.bolsista,
                ponto_virada=student.ponto_virada
            )
            
            # Rastrear predição
            metrics.track_prediction(
                prediction=result["risco_defasagem"],
                latency_ms=tracker.latency_ms
            )
            
            return PredictionOutput(
                risco_defasagem=result["risco_defasagem"],
                probabilidade=result["probabilidade"],
                nivel_risco=result["nivel_risco"],
                probabilidades_por_classe=result["probabilidades_por_classe"],
                timestamp=datetime.fromisoformat(result["timestamp"])
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            metrics.track_error(
                error_type="PredictionError",
                error_message=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erro ao realizar predição: {str(e)}"
            )


@router.post(
    "/batch",
    response_model=BatchPredictionOutput,
    summary="Predição em Lote",
    description="Realiza predição de risco para múltiplos estudantes",
    responses={
        200: {"description": "Predições realizadas com sucesso"},
        400: {"model": ErrorResponse, "description": "Dados inválidos"},
        503: {"model": ErrorResponse, "description": "Modelo não disponível"},
    }
)
async def predict_batch(batch: BatchPredictionInput) -> BatchPredictionOutput:
    """
    Realiza predição de risco de defasagem para múltiplos estudantes.
    
    Aceita até 1000 estudantes por requisição.
    Retorna lista de predições na mesma ordem dos dados de entrada.
    """
    metrics = get_metrics_tracker()
    start_time = time.perf_counter()
    
    try:
        predictor = get_predictor()
        predictions = []
        
        for student in batch.students:
            result = predictor.predict_risk(
                fase=student.fase,
                idade=student.idade,
                genero=student.genero,
                anos_na_pm=student.anos_na_pm,
                inde=student.inde,
                ian=student.ian,
                ida=student.ida,
                ieg=student.ieg,
                iaa=student.iaa,
                ips=student.ips,
                ipp=student.ipp,
                ipv=student.ipv,
                pedra=student.pedra,
                instituicao_ensino=student.instituicao_ensino,
                bolsista=student.bolsista,
                ponto_virada=student.ponto_virada
            )
            
            predictions.append(PredictionOutput(
                risco_defasagem=result["risco_defasagem"],
                probabilidade=result["probabilidade"],
                nivel_risco=result["nivel_risco"],
                probabilidades_por_classe=result["probabilidades_por_classe"],
                timestamp=datetime.fromisoformat(result["timestamp"])
            ))
            
            # Rastrear cada predição
            metrics.track_prediction(prediction=result["risco_defasagem"])
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Predição em lote concluída: {len(predictions)} estudantes "
            f"em {processing_time:.2f}ms"
        )
        
        return BatchPredictionOutput(
            predictions=predictions,
            total=len(predictions),
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        metrics.track_error(
            error_type="BatchPredictionError",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao realizar predição em lote: {str(e)}"
        )


@router.get(
    "/risk-levels",
    summary="Níveis de Risco",
    description="Retorna informações sobre os níveis de risco"
)
async def get_risk_levels() -> dict:
    """
    Retorna descrição dos níveis de risco.
    """
    return {
        "levels": [
            {
                "code": 0,
                "label": "BAIXO",
                "description": "Estudante com baixo risco de defasagem escolar",
                "criteria": [
                    "Sem defasagem",
                    "INDE >= 6.5",
                    "Indicadores estáveis ou crescentes"
                ]
            },
            {
                "code": 1,
                "label": "MÉDIO",
                "description": "Estudante com risco moderado de defasagem escolar",
                "criteria": [
                    "Defasagem de 1 ano",
                    "INDE entre 5.5 e 6.5",
                    "Tendência de queda nos indicadores"
                ]
            },
            {
                "code": 2,
                "label": "ALTO",
                "description": "Estudante com alto risco de defasagem escolar",
                "criteria": [
                    "Defasagem >= 2 anos",
                    "INDE < 5.5",
                    "Múltiplos indicadores baixos"
                ]
            }
        ]
    }
