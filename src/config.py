"""
Configurações e constantes do projeto Passos Mágicos ML.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from pydantic_settings import BaseSettings
from pydantic import Field


# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
MONITORING_DIR = PROJECT_ROOT / "monitoring"


class Settings(BaseSettings):
    """Configurações da aplicação."""
    
    # Ambiente
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Modelo
    model_path: str = Field(default="models/model.joblib", env="MODEL_PATH")
    model_version: str = Field(default="1.0.0", env="MODEL_VERSION")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    # Dados
    data_file: str = Field(
        default="data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
        env="DATA_FILE"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instância global de configurações
settings = Settings()


# Configurações dos indicadores
INDICADORES = {
    "INDE": "Índice de Desenvolvimento Educacional",
    "IAN": "Indicador de Adequação ao Nível",
    "IDA": "Indicador de Desempenho Acadêmico",
    "IEG": "Indicador de Engajamento",
    "IAA": "Indicador de Autoavaliação",
    "IPS": "Indicador Psicossocial",
    "IPP": "Indicador Psicopedagógico",
    "IPV": "Indicador de Ponto de Virada",
}

# Limites dos indicadores (escala 0-10)
INDICADOR_MIN = 0.0
INDICADOR_MAX = 10.0

# Classificação PEDRA baseada no INDE
PEDRA_LIMITES = {
    "Quartzo": (2.405, 5.506),
    "Ágata": (5.506, 6.868),
    "Ametista": (6.868, 8.230),
    "Topázio": (8.230, 9.294),
}

# Mapeamento de níveis de risco
RISCO_LABELS = {
    0: "BAIXO",
    1: "MÉDIO",
    2: "ALTO",
}

# Thresholds para classificação de risco
RISCO_THRESHOLDS = {
    "inde_alto_risco": 5.5,
    "inde_medio_risco": 6.5,
    "defasagem_alta": 2,
    "defasagem_media": 1,
}

# Colunas esperadas no dataset
COLUNAS_NUMERICAS: List[str] = [
    "INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV",
    "NOTA_PORT", "NOTA_MAT", "NOTA_ING",
]

COLUNAS_CATEGORICAS: List[str] = [
    "INSTITUICAO_ENSINO_ALUNO", "FASE", "TURMA", "GENERO", "PEDRA",
]

COLUNAS_BOOLEANAS: List[str] = [
    "PONTO_VIRADA", "BOLSISTA",
]

# Features para o modelo
FEATURES_MODELO: List[str] = [
    "FASE", "IDADE", "ANOS_PM", "INDE", "IAN", "IDA", "IEG", 
    "IAA", "IPS", "IPP", "IPV", "BOLSISTA", "PONTO_VIRADA",
]

# Sheets do Excel
SHEETS = ["PEDE_2022", "PEDE_2023", "PEDE_2024"]

# Configurações de treinamento
TRAIN_CONFIG: Dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "scoring": "recall_weighted",  # Prioridade: não perder alunos em risco (multiclasse)
}

# Configurações dos modelos
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "class_weight": ["balanced"],
        "max_iter": [1000],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "scale_pos_weight": [1, 2, 3],  # Para desbalanceamento
    },
}

# Métricas de avaliação
METRICAS_AVALIACAO: List[str] = [
    "recall",
    "precision",
    "f1",
    "roc_auc",
    "accuracy",
]
