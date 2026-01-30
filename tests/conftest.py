"""
Configurações e fixtures para testes pytest.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def sample_student_data() -> Dict:
    """Retorna dados de um estudante de exemplo."""
    return {
        "FASE": 5,
        "IDADE": 12,
        "GENERO": "M",
        "ANOS_PM": 2,
        "INDE": 6.5,
        "IAN": 7.0,
        "IDA": 6.0,
        "IEG": 7.5,
        "IAA": 7.0,
        "IPS": 6.5,
        "IPP": 7.0,
        "IPV": 6.0,
        "PEDRA": "Ágata",
        "INSTITUICAO_ENSINO_ALUNO": "Escola Municipal",
        "BOLSISTA": False,
        "PONTO_VIRADA": False,
        "ANO_PEDE": 2024
    }


@pytest.fixture
def sample_dataframe(sample_student_data) -> pd.DataFrame:
    """Retorna DataFrame com dados de exemplo."""
    # Criar 100 registros de exemplo
    data = []
    
    for i in range(100):
        record = sample_student_data.copy()
        record["RA"] = f"RA{i:04d}"
        record["FASE"] = np.random.randint(1, 9)
        record["IDADE"] = np.random.randint(7, 18)
        record["INDE"] = np.random.uniform(4, 9)
        record["IAN"] = np.random.uniform(4, 9)
        record["IDA"] = np.random.uniform(4, 9)
        record["IEG"] = np.random.uniform(4, 9)
        record["IAA"] = np.random.uniform(4, 9)
        record["IPS"] = np.random.uniform(4, 9)
        record["IPP"] = np.random.uniform(4, 9)
        record["IPV"] = np.random.uniform(4, 9)
        record["GENERO"] = np.random.choice(["M", "F"])
        record["BOLSISTA"] = np.random.choice([True, False])
        record["PONTO_VIRADA"] = np.random.choice([True, False])
        data.append(record)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_with_missing(sample_dataframe) -> pd.DataFrame:
    """Retorna DataFrame com valores faltantes."""
    df = sample_dataframe.copy()
    
    # Adicionar valores faltantes
    np.random.seed(42)
    for col in ["INDE", "IAN", "IDA"]:
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan
    
    return df


@pytest.fixture
def sample_features_df(sample_dataframe) -> pd.DataFrame:
    """Retorna DataFrame apenas com features numéricas."""
    numeric_cols = [
        "FASE", "IDADE", "INDE", "IAN", "IDA", "IEG", 
        "IAA", "IPS", "IPP", "IPV"
    ]
    return sample_dataframe[numeric_cols].copy()


@pytest.fixture
def sample_target(sample_dataframe) -> pd.Series:
    """Retorna Series com target de exemplo."""
    # Criar target baseado em INDE
    def calculate_risk(inde):
        if inde < 5.5:
            return 2  # Alto risco
        elif inde < 6.5:
            return 1  # Médio risco
        else:
            return 0  # Baixo risco
    
    return sample_dataframe["INDE"].apply(calculate_risk)


@pytest.fixture
def api_student_input() -> Dict:
    """Retorna dados de entrada para a API."""
    return {
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


@pytest.fixture
def temp_model_path(tmp_path) -> Path:
    """Retorna caminho temporário para modelo."""
    return tmp_path / "model.joblib"


@pytest.fixture
def mock_model():
    """Retorna modelo mock para testes."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    # Treinar com dados sintéticos
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    pipeline.fit(X, y)
    
    return pipeline


# Configurações de ambiente para testes
@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Configura ambiente de testes."""
    # Usar diretórios temporários
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "test.log"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
