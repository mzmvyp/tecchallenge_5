# PROMPT COMPLETO PARA MELHORAR O tech_henrique

> Cole este prompt inteiro no Gemini para ele executar as correções no projeto tech_henrique.

---

## PROMPT INÍCIO

Você é um engenheiro de ML sênior. Preciso que você faça uma série de melhorias no meu projeto `tech_henrique` (Datathon Passos Mágicos) para que ele atenda 100% dos requisitos do Tech Challenge. O projeto já tem uma base funcional com FastAPI, RandomForest, Docker Compose com Prometheus/Grafana/MLflow/Loki, mas precisa de melhorias significativas em documentação, modularização, feature engineering, testes e API.

Faça TODAS as alterações abaixo de uma vez. Não pule nenhuma seção.

---

## 1. DOCUMENTAÇÃO (PRIORIDADE MÁXIMA - hoje temos 2 linhas de README)

### 1.1 Criar README.md completo (mínimo 300 linhas)

```markdown
# Passos Mágicos - Predição de Risco de Evasão Escolar

## Sobre o Projeto
- Explicar o contexto da ONG Passos Mágicos
- Explicar o problema: prever risco de defasagem/evasão dos alunos
- Explicar a solução: modelo de ML que classifica alunos em níveis de risco
- Explicar os níveis de risco: BAIXO, MÉDIO, ALTO (ou a classificação usada)

## Stack Tecnológica
- Python 3.11+
- FastAPI (API REST)
- scikit-learn (ML)
- MLflow (versionamento de modelos)
- Prometheus + Grafana (monitoramento)
- Loki + Promtail (logs centralizados)
- Docker + Docker Compose (deploy)
- pytest (testes)

## Estrutura do Projeto
(colocar árvore completa de diretórios com explicação de cada pasta)

## Como Instalar
- Pré-requisitos
- Clone do repositório
- Criar ambiente virtual: python -m venv venv && source venv/bin/activate
- Instalar dependências: pip install -r requirements.txt
- Configurar variáveis de ambiente (.env)

## Como Treinar o Modelo
- Comando para treinar
- Explicar os dados de entrada (PEDE 2022, 2023, 2024)
- Explicar os indicadores: INDE, IAN, IDA, IEG, IAA, IPS, IPP, IPV
- Explicar o que é PEDRA (classificação por faixa do INDE)

## Como Rodar a API
- Modo desenvolvimento: uvicorn app.main:app --reload
- Modo Docker: docker-compose up -d
- Acessar docs: http://localhost:8000/docs

## Endpoints da API
- Listar TODOS os endpoints com método HTTP, URL, descrição, exemplo de request/response

## Monitoramento
- Como acessar Grafana (porta)
- Como acessar Prometheus (porta)
- Como acessar MLflow (porta)
- Dashboards disponíveis

## Testes
- Como rodar: pytest
- Como ver cobertura: pytest --cov=app --cov=src --cov-report=html
- Meta de cobertura: 80%+

## Pipeline de ML
- Explicar cada etapa: coleta → limpeza → feature engineering → treino → avaliação → deploy
- Métricas usadas: accuracy, precision, recall, f1-score, classification_report
- Algoritmos testados

## Contribuição
- Como contribuir, padrões de código
```

### 1.2 Criar arquivo docs/RELATORIO_SISTEMA.md

Criar um relatório técnico completo com:
- Introdução e contexto do problema educacional
- Metodologia (CRISP-DM ou similar)
- Análise exploratória dos dados (descrever distribuições, correlações, insights)
- Descrição do feature engineering aplicado
- Comparação de modelos (se testar mais de um)
- Resultados e métricas finais
- Conclusões e próximos passos
- Mínimo 200 linhas

### 1.3 Criar docs/api_reference.md

Documentação completa de cada endpoint:
- URL, método, descrição
- Parâmetros de entrada com tipos e validações
- Exemplo de request (curl e Python)
- Exemplo de response (JSON)
- Códigos de erro possíveis

### 1.4 Criar docs/deployment.md

Guia de deploy:
- Deploy local com Docker Compose
- Variáveis de ambiente necessárias
- Como escalar
- Health checks
- Troubleshooting

### 1.5 Criar .env.example

```env
# Ambiente
ENVIRONMENT=development
DEBUG=True

# Modelo
MODEL_PATH=app/model/modelo.pkl
MODEL_VERSION=1.0.0

# API
API_HOST=0.0.0.0
API_PORT=8000

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5050

# Logging
LOG_LEVEL=INFO
```

---

## 2. MODULARIZAÇÃO DO CÓDIGO

### 2.1 Reorganizar src/ em pacotes Python

Transformar os arquivos de `src/` em pacotes com `__init__.py`:

```
src/
├── __init__.py
├── config.py                    ← CRIAR: configurações centralizadas
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py           ← mover lógica de carregar CSVs
│   ├── data_cleaner.py          ← mover lógica de limpeza
│   └── data_transformer.py      ← mover lógica de transformação
├── features/
│   ├── __init__.py
│   ├── feature_engineering.py   ← mover/expandir feature engineering
│   └── feature_selection.py     ← CRIAR: seleção de features por importância
├── models/
│   ├── __init__.py
│   ├── train.py                 ← mover lógica de treino
│   ├── evaluate.py              ← mover lógica de avaliação
│   └── predict.py               ← CRIAR: interface de predição
├── monitoring/
│   ├── __init__.py
│   ├── drift_detector.py        ← CRIAR: detecção de drift com Evidently ou manual
│   └── metrics_tracker.py       ← CRIAR: rastreador de métricas em memória
└── utils/
    ├── __init__.py
    ├── logger.py                ← CRIAR: configuração de logging estruturado
    └── helpers.py               ← mover funções utilitárias
```

### 2.2 Criar src/config.py centralizado

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "files"
    MODEL_DIR: Path = PROJECT_ROOT / "app" / "model"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Model
    MODEL_PATH: str = "app/model/modelo.pkl"
    MODEL_VERSION: str = "1.0.0"

    # Indicadores educacionais
    INDICADORES: list = ["INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"]

    # Classificação PEDRA por faixas de INDE
    PEDRA_LIMITS: dict = {
        "Quartzo": (0, 2.405),
        "Ágata": (2.405, 5.506),
        "Ametista": (5.506, 7.607),
        "Topázio": (7.607, 10.0)
    }

    # Classificação de risco
    RISK_LABELS: dict = {0: "BAIXO", 1: "MÉDIO", 2: "ALTO"}

    # Training
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    RANDOM_STATE: int = 42
    SCORING: str = "recall_weighted"

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2.3 Criar src/utils/logger.py

```python
import logging
import sys
from pathlib import Path

def setup_logger(name: str = "passos_magicos", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(console)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    ))
    logger.addHandler(file_handler)

    return logger
```

---

## 3. FEATURE ENGINEERING AVANÇADO

### 3.1 Expandir feature_engineering.py

Adicionar TODAS estas features ao pipeline de feature engineering:

```python
class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def create_all_features(self):
        self._ensure_numeric_columns()
        self._create_defasagem_features()
        self._create_temporal_features()
        self._create_indicator_aggregations()
        self._create_interaction_features()
        self._create_risk_target()
        return self.df

    def _ensure_numeric_columns(self):
        """Garantir que colunas de indicadores são numéricas"""
        indicadores = ["INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"]
        for col in indicadores:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def _create_defasagem_features(self):
        """Calcular defasagem escolar"""
        # FASE ideal vs FASE real baseado na idade
        if "IDADE" in self.df.columns and "FASE" in self.df.columns:
            self.df["FASE_IDEAL"] = self.df["IDADE"] - 5  # fase ideal = idade - 5
            self.df["DEFASAGEM"] = self.df["FASE_IDEAL"] - self.df["FASE"]
            self.df["TEM_DEFASAGEM"] = (self.df["DEFASAGEM"] > 0).astype(int)

    def _create_temporal_features(self):
        """Features temporais"""
        if "ANO_INGRESSO" in self.df.columns:
            self.df["ANOS_PM"] = 2024 - self.df["ANO_INGRESSO"]
            self.df["VETERANO"] = (self.df["ANOS_PM"] > 1).astype(int)

    def _create_indicator_aggregations(self):
        """Agregações estatísticas dos indicadores"""
        indicadores = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"]
        cols = [c for c in indicadores if c in self.df.columns]

        if cols:
            self.df["MEDIA_INDICADORES"] = self.df[cols].mean(axis=1)
            self.df["STD_INDICADORES"] = self.df[cols].std(axis=1)
            self.df["MIN_INDICADORES"] = self.df[cols].min(axis=1)
            self.df["MAX_INDICADORES"] = self.df[cols].max(axis=1)
            self.df["RANGE_INDICADORES"] = self.df["MAX_INDICADORES"] - self.df["MIN_INDICADORES"]
            # Coeficiente de variação
            self.df["CV_INDICADORES"] = self.df["STD_INDICADORES"] / (self.df["MEDIA_INDICADORES"] + 1e-8)

    def _create_interaction_features(self):
        """Features de interação entre indicadores"""
        if "IDA" in self.df.columns and "IEG" in self.df.columns:
            self.df["RATIO_IDA_IEG"] = self.df["IDA"] / (self.df["IEG"] + 1e-8)
        if "IPP" in self.df.columns and "IPS" in self.df.columns:
            self.df["RATIO_IPP_IPS"] = self.df["IPP"] / (self.df["IPS"] + 1e-8)
        if "IEG" in self.df.columns and "IDA" in self.df.columns:
            self.df["ENGAJ_X_DESEMP"] = self.df["IEG"] * self.df["IDA"]
        if "IAA" in self.df.columns and "IAN" in self.df.columns:
            self.df["AUTO_X_ALFABETIZACAO"] = self.df["IAA"] * self.df["IAN"]

    def _create_risk_target(self):
        """Criar target de risco multi-classe (3 níveis)"""
        # Se atualmente é binário (0/1), converter para 3 classes:
        # BAIXO=0: INDE alto, sem defasagem, indicadores bons
        # MÉDIO=1: INDE médio OU algum indicador baixo
        # ALTO=2: INDE baixo OU defasagem grande OU múltiplos indicadores baixos

        if "INDE" in self.df.columns:
            conditions = [
                self.df["INDE"] >= 7.0,   # BAIXO risco
                self.df["INDE"] >= 4.0,   # MÉDIO risco
                self.df["INDE"] < 4.0     # ALTO risco
            ]
            choices = [0, 1, 2]
            self.df["RISCO_DEFASAGEM"] = np.select(conditions, choices, default=1)
```

IMPORTANTE: Se o target atual é binário (0/1), converter para 3 classes (BAIXO/MÉDIO/ALTO). Se não for possível mudar o target, pelo menos adicionar todas as features de agregação e interação acima.

---

## 4. MÚLTIPLOS ALGORITMOS DE ML

### 4.1 Expandir o treinamento para testar 5 algoritmos

O projeto atualmente usa apenas RandomForest. Modificar o train.py para testar:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Tentar importar XGBoost (opcional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

MODEL_CONFIGS = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"]
        }
    }
}

if HAS_XGBOOST:
    MODEL_CONFIGS["XGBoost"] = {
        "model": XGBClassifier(random_state=42, eval_metric="mlogloss"),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2]
        }
    }
```

### 4.2 Usar SMOTE para balanceamento de classes

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# No treinamento:
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### 4.3 Usar StratifiedKFold com 5 folds

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    pipeline, param_grid, cv=cv,
    scoring="recall_weighted",  # priorizar encontrar alunos em risco
    n_jobs=-1, verbose=1
)
```

---

## 5. API - NOVOS ENDPOINTS

### 5.1 Adicionar endpoint de predição em lote (batch)

```python
from pydantic import BaseModel, Field
from typing import List
import time

class BatchPredictionInput(BaseModel):
    students: List[StudentInput] = Field(..., min_length=1, max_length=1000)

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total: int
    processing_time_ms: float

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchPredictionInput):
    start = time.time()
    predictions = []
    for student in batch.students:
        # reutilizar lógica do /predict individual
        pred = make_prediction(student)
        predictions.append(pred)
    elapsed = (time.time() - start) * 1000
    return BatchPredictionOutput(
        predictions=predictions,
        total=len(predictions),
        processing_time_ms=round(elapsed, 2)
    )
```

### 5.2 Adicionar endpoints de health check

```python
from datetime import datetime

START_TIME = datetime.utcnow()

@app.get("/health")
async def health():
    uptime = (datetime.utcnow() - START_TIME).total_seconds()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0",
        "uptime_seconds": round(uptime, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}
```

### 5.3 Adicionar endpoint de informações do modelo

```python
@app.get("/health/model")
async def model_info():
    return {
        "model_type": type(model).__name__ if model else None,
        "version": "1.0.0",
        "features": list(FEATURE_NAMES),
        "n_features": len(FEATURE_NAMES),
        "risk_levels": ["BAIXO", "MÉDIO", "ALTO"],
        "model_path": "app/model/modelo.pkl"
    }
```

### 5.4 Adicionar endpoint de níveis de risco

```python
@app.get("/predict/risk-levels")
async def risk_levels():
    return {
        "levels": {
            "BAIXO": "Aluno com bom desempenho, sem defasagem significativa",
            "MÉDIO": "Aluno com desempenho mediano, necessita atenção",
            "ALTO": "Aluno em risco de evasão, necessita intervenção urgente"
        }
    }
```

### 5.5 Melhorar schemas Pydantic com validações e descrições

```python
from pydantic import BaseModel, Field, field_validator

class StudentInput(BaseModel):
    fase: int = Field(..., ge=1, le=8, description="Fase/série do aluno (1 a 8)")
    idade: int = Field(..., ge=6, le=25, description="Idade do aluno")
    genero: str = Field(..., description="Gênero: M ou F")

    # Indicadores (0 a 10)
    inde: float = Field(..., ge=0, le=10, description="Índice de Desenvolvimento Educacional")
    ian: float = Field(..., ge=0, le=10, description="Indicador de Adequação de Nível")
    ida: float = Field(..., ge=0, le=10, description="Indicador de Desempenho Acadêmico")
    ieg: float = Field(..., ge=0, le=10, description="Indicador de Engajamento")
    iaa: float = Field(..., ge=0, le=10, description="Indicador de Autoavaliação")
    ips: float = Field(..., ge=0, le=10, description="Indicador Psicossocial")
    ipp: float = Field(..., ge=0, le=10, description="Indicador Psicopedagógico")
    ipv: float = Field(..., ge=0, le=10, description="Indicador do Ponto de Virada")

    pedra: str = Field(..., description="Classificação PEDRA: Quartzo, Ágata, Ametista ou Topázio")
    bolsista: bool = Field(False, description="Se o aluno é bolsista")
    ponto_virada: bool = Field(False, description="Se atingiu ponto de virada")

    @field_validator("genero")
    @classmethod
    def normalize_genero(cls, v):
        v = v.strip().upper()
        if v not in ("M", "F"):
            raise ValueError("Gênero deve ser M ou F")
        return v

    @field_validator("pedra")
    @classmethod
    def normalize_pedra(cls, v):
        valid = ["quartzo", "ágata", "agata", "ametista", "topázio", "topazio"]
        if v.strip().lower() not in valid:
            raise ValueError(f"PEDRA deve ser uma de: Quartzo, Ágata, Ametista, Topázio")
        return v.strip().capitalize()

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "fase": 4,
                "idade": 12,
                "genero": "M",
                "inde": 6.5,
                "ian": 7.0,
                "ida": 6.8,
                "ieg": 7.2,
                "iaa": 6.0,
                "ips": 7.5,
                "ipp": 6.3,
                "ipv": 5.8,
                "pedra": "Ametista",
                "bolsista": False,
                "ponto_virada": False
            }]
        }
    }

class PredictionOutput(BaseModel):
    risco_defasagem: int = Field(..., description="Classificação numérica: 0=BAIXO, 1=MÉDIO, 2=ALTO")
    probabilidade: float = Field(..., description="Probabilidade da classe predita")
    nivel_risco: str = Field(..., description="Nível de risco: BAIXO, MÉDIO ou ALTO")
    probabilidades_por_classe: dict = Field(..., description="Probabilidades para cada classe")
    timestamp: str = Field(..., description="Momento da predição")
```

---

## 6. DOCKERFILE MELHORADO

Substituir o Dockerfile atual (6 linhas) por um production-ready:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar diretórios necessários
RUN mkdir -p logs monitoring/reports

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. MAKEFILE

Criar Makefile com comandos de automação:

```makefile
.PHONY: help install install-dev run run-dev test test-cov train lint format clean docker-build docker-run docker-stop docs

help:  ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependências de produção
	pip install -r requirements.txt

install-dev:  ## Instala dependências de desenvolvimento
	pip install -r requirements.txt
	pip install -r requirements-dev.txt 2>/dev/null || true
	pip install pytest pytest-cov pytest-asyncio httpx black isort flake8 mypy

run:  ## Roda a API em produção
	uvicorn app.main:app --host 0.0.0.0 --port 8000

run-dev:  ## Roda a API em modo desenvolvimento
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:  ## Roda os testes
	pytest tests/ -v --tb=short

test-cov:  ## Roda testes com cobertura (meta 80%)
	pytest tests/ -v --cov=app --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

train:  ## Treina o modelo
	python -m src.models.train

lint:  ## Verifica qualidade do código
	flake8 src/ app/ --max-line-length=120
	mypy src/ app/ --ignore-missing-imports || true

format:  ## Formata o código
	black src/ app/ tests/ --line-length=120
	isort src/ app/ tests/

clean:  ## Limpa arquivos temporários
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

docker-build:  ## Build da imagem Docker
	docker-compose build

docker-run:  ## Sobe todos os serviços
	docker-compose up -d

docker-stop:  ## Para todos os serviços
	docker-compose down
```

---

## 8. TESTES - EXPANDIR COBERTURA

### 8.1 Criar/expandir conftest.py com fixtures compartilhadas

```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_student_data():
    """Dados de um aluno para testes"""
    return {
        "fase": 4,
        "idade": 12,
        "genero": "M",
        "inde": 6.5,
        "ian": 7.0,
        "ida": 6.8,
        "ieg": 7.2,
        "iaa": 6.0,
        "ips": 7.5,
        "ipp": 6.3,
        "ipv": 5.8,
        "pedra": "Ametista",
        "bolsista": False,
        "ponto_virada": False
    }

@pytest.fixture
def sample_dataframe():
    """DataFrame de exemplo para testes de pipeline"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "NOME": [f"Aluno_{i}" for i in range(n)],
        "FASE": np.random.randint(1, 9, n),
        "IDADE": np.random.randint(8, 18, n),
        "GENERO": np.random.choice(["M", "F"], n),
        "INDE": np.random.uniform(2, 9, n).round(2),
        "IAN": np.random.uniform(0, 10, n).round(2),
        "IDA": np.random.uniform(0, 10, n).round(2),
        "IEG": np.random.uniform(0, 10, n).round(2),
        "IAA": np.random.uniform(0, 10, n).round(2),
        "IPS": np.random.uniform(0, 10, n).round(2),
        "IPP": np.random.uniform(0, 10, n).round(2),
        "IPV": np.random.uniform(0, 10, n).round(2),
        "PEDRA": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
        "BOLSISTA": np.random.choice([True, False], n),
        "PONTO_VIRADA": np.random.choice([True, False], n),
    })

@pytest.fixture
def sample_batch_data(sample_student_data):
    """Batch de alunos para testes"""
    return {"students": [sample_student_data] * 5}
```

### 8.2 Expandir test_api.py

Adicionar testes para os NOVOS endpoints:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "model_loaded" in data

def test_health_ready():
    response = client.get("/health/ready")
    assert response.status_code in [200, 503]

def test_health_live():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_model_info():
    response = client.get("/health/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features" in data

def test_predict_single(sample_student_data):
    response = client.post("/predict", json=sample_student_data)
    assert response.status_code == 200
    data = response.json()
    assert "risco_defasagem" in data or "prediction" in data
    assert "probabilidade" in data or "probability" in data

def test_predict_batch(sample_batch_data):
    response = client.post("/predict/batch", json=sample_batch_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total" in data
    assert data["total"] == 5
    assert "processing_time_ms" in data

def test_predict_invalid_data():
    response = client.post("/predict", json={"fase": -1})
    assert response.status_code == 422

def test_predict_batch_empty():
    response = client.post("/predict/batch", json={"students": []})
    assert response.status_code == 422

def test_risk_levels():
    response = client.get("/predict/risk-levels")
    assert response.status_code == 200
    data = response.json()
    assert "levels" in data

def test_root():
    response = client.get("/")
    assert response.status_code == 200
```

### 8.3 Criar test_feature_engineering.py (se não existir completo)

```python
def test_indicator_aggregations(sample_dataframe):
    fe = FeatureEngineer(sample_dataframe)
    result = fe.create_all_features()
    assert "MEDIA_INDICADORES" in result.columns
    assert "STD_INDICADORES" in result.columns
    assert "MIN_INDICADORES" in result.columns
    assert "MAX_INDICADORES" in result.columns
    assert "RANGE_INDICADORES" in result.columns
    assert "CV_INDICADORES" in result.columns

def test_interaction_features(sample_dataframe):
    fe = FeatureEngineer(sample_dataframe)
    result = fe.create_all_features()
    assert "RATIO_IDA_IEG" in result.columns
    assert "ENGAJ_X_DESEMP" in result.columns

def test_defasagem_features(sample_dataframe):
    fe = FeatureEngineer(sample_dataframe)
    result = fe.create_all_features()
    assert "DEFASAGEM" in result.columns
    assert "TEM_DEFASAGEM" in result.columns
```

### 8.4 Configurar cobertura mínima de 80% no pyproject.toml

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src", "app"]
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

---

## 9. MIDDLEWARE E TRATAMENTO DE ERROS NA API

### 9.1 Adicionar logging middleware

```python
# app/middleware/logging_middleware.py
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("passos_magicos")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration:.2f}ms"
        )
        return response
```

### 9.2 Adicionar exception handler global

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

### 9.3 Adicionar CORS middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restringir em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 10. pyproject.toml COMPLETO

Se não existir, criar. Se existir, completar:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "passos-magicos-ml"
version = "1.0.0"
description = "Sistema de predição de risco de evasão escolar - Datathon Passos Mágicos"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Henrique", email = "henrique@example.com"}
]
keywords = ["machine-learning", "education", "prediction", "fastapi"]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "httpx>=0.24",
    "black>=23.0",
    "isort>=5.12",
    "flake8>=6.0",
    "mypy>=1.0",
]

[tool.black]
line-length = 120
target-version = ["py311"]

[tool.isort]
profile = "black"
line_length = 120

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.coverage.run]
source = ["src", "app"]
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

---

## 11. MkDocs (DOCUMENTAÇÃO COMO SITE)

### 11.1 Criar mkdocs.yml

```yaml
site_name: Passos Mágicos ML - Documentação
site_description: Sistema de Predição de Risco Educacional
site_author: Henrique

theme:
  name: material
  language: pt-BR
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      toggle:
        icon: material/brightness-7
        name: Modo escuro
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      toggle:
        icon: material/brightness-4
        name: Modo claro
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - search.suggest
    - content.code.copy

nav:
  - Home: index.md
  - API Reference: api_reference.md
  - Pipeline ML: pipeline.md
  - Deploy: deployment.md
  - Relatório: RELATORIO_SISTEMA.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src, app]

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.tabbed
  - admonition
  - toc:
      permalink: true
```

### 11.2 Adicionar dependências do MkDocs no requirements

```
mkdocs>=1.5
mkdocs-material>=9.0
mkdocstrings[python]>=0.22
```

---

## 12. CHECKLIST FINAL DE VALIDAÇÃO

Depois de fazer todas as alterações, verifique:

- [ ] README.md tem mais de 300 linhas e cobre TODOS os tópicos listados
- [ ] Existe docs/RELATORIO_SISTEMA.md com relatório técnico completo
- [ ] Existe docs/api_reference.md com documentação de todos os endpoints
- [ ] Existe docs/deployment.md com guia de deploy
- [ ] Existe .env.example
- [ ] src/ está organizado em pacotes com __init__.py
- [ ] Existe src/config.py centralizado
- [ ] Existe src/utils/logger.py
- [ ] Feature engineering tem agregações (MEDIA, STD, MIN, MAX, RANGE, CV)
- [ ] Feature engineering tem features de interação (ratios, multiplicações)
- [ ] Pipeline testa pelo menos 3 algoritmos (não só RandomForest)
- [ ] Usa StratifiedKFold com 5 folds
- [ ] API tem endpoint /predict/batch
- [ ] API tem endpoints /health, /health/ready, /health/live
- [ ] API tem /health/model e /predict/risk-levels
- [ ] Schemas Pydantic têm validações com Field(..., ge=, le=, description=)
- [ ] Dockerfile tem HEALTHCHECK
- [ ] Makefile existe com pelo menos 10 comandos
- [ ] pyproject.toml completo com configs de pytest, coverage, black, isort
- [ ] mkdocs.yml configurado
- [ ] Testes cobrem os novos endpoints (batch, health, risk-levels)
- [ ] Testes cobrem feature engineering
- [ ] pytest --cov retorna 80%+ de cobertura
- [ ] Middleware de logging está configurado
- [ ] Exception handler global está configurado
- [ ] CORS middleware está configurado
- [ ] O código roda sem erros: uvicorn app.main:app
- [ ] Os testes passam: pytest tests/ -v

## IMPORTANTE

- NÃO remova a stack existente de monitoramento (Prometheus, Grafana, MLflow, Loki) - ela é um PONTO FORTE do projeto
- NÃO remova os endpoints /reload e /retrain - eles são diferenciais
- MANTENHA a compatibilidade com o docker-compose.yml existente (7 serviços)
- ADICIONE os novos endpoints SEM quebrar os existentes
- Se houver conflito entre a estrutura existente e a proposta, ADAPTE a proposta para funcionar com o que já existe

---

## FIM DO PROMPT
