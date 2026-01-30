"""
Schemas Pydantic para a API Passos Mágicos.
"""

from datetime import datetime
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field


def normalize_genero(v: str) -> str:
    """Normaliza gênero para uppercase e valida."""
    if isinstance(v, str):
        v = v.upper()
        if v not in ("M", "F"):
            raise ValueError("Gênero deve ser 'M' ou 'F'")
    return v


def normalize_pedra(v: str) -> str:
    """Normaliza pedra para title case e valida."""
    if isinstance(v, str):
        v = v.title()
        valid_pedras = {"Quartzo", "Ágata", "Ametista", "Topázio"}
        if v not in valid_pedras:
            raise ValueError(f"Pedra deve ser uma de: {', '.join(valid_pedras)}")
    return v


# Tipos customizados com validação antes da conversão
GeneroType = Annotated[str, BeforeValidator(normalize_genero)]
PedraType = Annotated[str, BeforeValidator(normalize_pedra)]


class StudentInput(BaseModel):
    """Schema de entrada para dados de um estudante."""
    
    fase: int = Field(..., ge=1, le=8, description="Fase atual do estudante na Passos Mágicos (1-8)")
    idade: int = Field(..., ge=6, le=25, description="Idade do estudante")
    genero: GeneroType = Field(..., description="Gênero do estudante (M ou F)")
    anos_na_pm: int = Field(..., ge=0, le=15, description="Anos na Passos Mágicos")
    inde: float = Field(..., ge=0, le=10, description="Índice de Desenvolvimento Educacional")
    ian: float = Field(..., ge=0, le=10, description="Indicador de Adequação ao Nível")
    ida: float = Field(..., ge=0, le=10, description="Indicador de Desempenho Acadêmico")
    ieg: float = Field(..., ge=0, le=10, description="Indicador de Engajamento")
    iaa: float = Field(..., ge=0, le=10, description="Indicador de Autoavaliação")
    ips: float = Field(..., ge=0, le=10, description="Indicador Psicossocial")
    ipp: float = Field(..., ge=0, le=10, description="Indicador Psicopedagógico")
    ipv: float = Field(..., ge=0, le=10, description="Indicador de Ponto de Virada")
    pedra: PedraType = Field(..., description="Classificação PEDRA baseada no INDE")
    instituicao_ensino: str = Field(..., min_length=1, description="Instituição de ensino do aluno")
    bolsista: bool = Field(..., description="Se o estudante é bolsista")
    ponto_virada: bool = Field(default=False, description="Se atingiu ponto de virada")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "fase": 5,
                "idade": 12,
                "genero": "M",
                "anos_na_pm": 2,
                "inde": 6.5,
                "ian": 7.0,
                "ida": 6.0,
                "ieg": 7.5,
                "iaa": 7.0,
                "ips": 6.5,
                "ipp": 7.0,
                "ipv": 6.0,
                "pedra": "Ágata",
                "instituicao_ensino": "Escola Municipal",
                "bolsista": False,
                "ponto_virada": False
            }
        }
    }


class PredictionOutput(BaseModel):
    """Schema de saída para predição."""
    
    risco_defasagem: int = Field(..., ge=0, le=2, description="Código de risco (0=Baixo, 1=Médio, 2=Alto)")
    probabilidade: float = Field(..., ge=0, le=1, description="Probabilidade da classe predita")
    nivel_risco: Literal["BAIXO", "MÉDIO", "ALTO"] = Field(..., description="Nível de risco textual")
    probabilidades_por_classe: Dict[str, float] = Field(
        ..., description="Probabilidades para cada classe de risco"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da predição")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "risco_defasagem": 1,
                "probabilidade": 0.65,
                "nivel_risco": "MÉDIO",
                "probabilidades_por_classe": {
                    "BAIXO": 0.25,
                    "MÉDIO": 0.65,
                    "ALTO": 0.10
                },
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    }


class BatchPredictionInput(BaseModel):
    """Schema para predição em lote."""
    
    students: List[StudentInput] = Field(..., min_length=1, max_length=1000)


class BatchPredictionOutput(BaseModel):
    """Schema de saída para predição em lote."""
    
    predictions: List[PredictionOutput]
    total: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    
    status: Literal["healthy", "unhealthy"] = "healthy"
    model_loaded: bool = True
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class MetricsResponse(BaseModel):
    """Schema para resposta de métricas."""
    
    total_predictions: int
    total_errors: int
    error_rate: float
    prediction_distribution: Dict[str, int]
    latency: Optional[Dict[str, float]] = None
    predictions_per_minute: float
    session_uptime_seconds: float


class DriftReportResponse(BaseModel):
    """Schema para resposta de relatório de drift."""
    
    drift_detected: bool
    features_with_drift: List[Dict[str, float]]
    overall_drift_score: float
    report_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Schema para respostas de erro."""
    
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "Validation Error",
                "detail": "Campo 'idade' deve ser um número entre 6 e 25",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """Schema para informações do modelo."""
    
    model_type: str
    version: str
    model_path: str
    features: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None
