"""
Funções auxiliares para o projeto Passos Mágicos ML.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Garante que um diretório existe, criando-o se necessário.
    
    Args:
        path: Caminho do diretório
    
    Returns:
        Path do diretório
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carrega configurações de um arquivo JSON.
    
    Args:
        config_path: Caminho para o arquivo de configuração
    
    Returns:
        Dicionário com as configurações
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Salva configurações em um arquivo JSON.
    
    Args:
        config: Dicionário com as configurações
        config_path: Caminho para salvar o arquivo
    """
    config_path = Path(config_path)
    ensure_dir(config_path.parent)
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_model(model: Any, path: Union[str, Path], compress: int = 3) -> None:
    """
    Salva um modelo usando joblib.
    
    Args:
        model: Modelo a ser salvo
        path: Caminho para salvar o modelo
        compress: Nível de compressão (0-9)
    """
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path, compress=compress)


def load_model(path: Union[str, Path]) -> Any:
    """
    Carrega um modelo usando joblib.
    
    Args:
        path: Caminho do modelo
    
    Returns:
        Modelo carregado
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    
    return joblib.load(path)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_error: bool = True
) -> bool:
    """
    Valida se um DataFrame possui as colunas necessárias.
    
    Args:
        df: DataFrame a ser validado
        required_columns: Lista de colunas obrigatórias
        raise_error: Se deve levantar erro quando colunas faltarem
    
    Returns:
        True se todas as colunas existem
    
    Raises:
        ValueError: Se colunas estão faltando e raise_error=True
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        if raise_error:
            raise ValueError(
                f"Colunas obrigatórias faltando no DataFrame: {missing_columns}"
            )
        return False
    
    return True


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e padroniza nomes de colunas.
    
    Args:
        df: DataFrame original
    
    Returns:
        DataFrame com colunas padronizadas
    """
    df = df.copy()
    
    # Remover espaços e caracteres especiais
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("[", "")
        .str.replace("]", "")
    )
    
    return df


def calculate_age_from_birth_year(
    birth_year: Union[int, pd.Series],
    reference_year: Optional[int] = None
) -> Union[int, pd.Series]:
    """
    Calcula idade a partir do ano de nascimento.
    
    Args:
        birth_year: Ano de nascimento
        reference_year: Ano de referência (padrão: ano atual)
    
    Returns:
        Idade calculada
    """
    from datetime import datetime
    
    if reference_year is None:
        reference_year = datetime.now().year
    
    return reference_year - birth_year


def categorize_inde(inde: float) -> str:
    """
    Categoriza INDE em classe PEDRA.
    
    Args:
        inde: Valor do INDE
    
    Returns:
        Categoria PEDRA (Quartzo, Ágata, Ametista, Topázio)
    """
    if pd.isna(inde):
        return "Desconhecido"
    
    if inde < 5.506:
        return "Quartzo"
    elif inde < 6.868:
        return "Ágata"
    elif inde < 8.230:
        return "Ametista"
    else:
        return "Topázio"


def calculate_defasagem(fase_atual: int, fase_ideal: int) -> int:
    """
    Calcula defasagem escolar.
    
    Args:
        fase_atual: Fase atual do aluno
        fase_ideal: Fase ideal baseada na idade
    
    Returns:
        Defasagem (negativo = atrasado, positivo = adiantado)
    """
    return fase_ideal - fase_atual


def get_risk_level(risco: int) -> str:
    """
    Converte código de risco para label.
    
    Args:
        risco: Código de risco (0, 1, 2)
    
    Returns:
        Label do nível de risco
    """
    labels = {
        0: "BAIXO",
        1: "MÉDIO",
        2: "ALTO"
    }
    return labels.get(risco, "DESCONHECIDO")


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formata valor como porcentagem.
    
    Args:
        value: Valor decimal (0-1)
        decimals: Número de casas decimais
    
    Returns:
        String formatada como porcentagem
    """
    return f"{value * 100:.{decimals}f}%"


def safe_divide(
    numerator: Union[float, np.ndarray, pd.Series],
    denominator: Union[float, np.ndarray, pd.Series],
    fill_value: float = 0.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Divisão segura que lida com divisão por zero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        fill_value: Valor para preencher quando denominador é zero
    
    Returns:
        Resultado da divisão
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        result = np.where(
            denominator != 0,
            numerator / denominator,
            fill_value
        )
        if isinstance(denominator, pd.Series):
            return pd.Series(result, index=denominator.index)
        return result
    else:
        return numerator / denominator if denominator != 0 else fill_value


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Retorna uso de memória de um DataFrame formatado.
    
    Args:
        df: DataFrame
    
    Returns:
        String com uso de memória
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    if memory_bytes < 1024:
        return f"{memory_bytes} bytes"
    elif memory_bytes < 1024 ** 2:
        return f"{memory_bytes / 1024:.2f} KB"
    elif memory_bytes < 1024 ** 3:
        return f"{memory_bytes / 1024 ** 2:.2f} MB"
    else:
        return f"{memory_bytes / 1024 ** 3:.2f} GB"
