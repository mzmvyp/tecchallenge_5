"""
Testes para o módulo de utilidades.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.helpers import (
    ensure_dir,
    load_config,
    save_config,
    save_model,
    load_model,
    validate_dataframe,
    clean_column_names,
    categorize_inde,
    calculate_defasagem,
    get_risk_level,
    safe_divide,
    get_memory_usage,
)
from src.utils.logger import setup_logger, get_logger, LoggerContext


class TestHelpers:
    """Testes para funções auxiliares."""
    
    def test_ensure_dir_creates_directory(self, tmp_path):
        """Testa criação de diretório."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        
        result = ensure_dir(new_dir)
        
        assert result.exists()
        assert result.is_dir()
    
    def test_ensure_dir_existing_directory(self, tmp_path):
        """Testa com diretório existente."""
        result = ensure_dir(tmp_path)
        
        assert result == tmp_path
        assert result.exists()
    
    def test_save_and_load_config(self, tmp_path):
        """Testa salvar e carregar configuração."""
        config = {"key": "value", "number": 42, "list": [1, 2, 3]}
        config_path = tmp_path / "config.json"
        
        save_config(config, config_path)
        loaded = load_config(config_path)
        
        assert loaded == config
    
    def test_load_config_not_found(self, tmp_path):
        """Testa carregar config inexistente."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")
    
    def test_save_and_load_model(self, tmp_path, mock_model):
        """Testa salvar e carregar modelo."""
        model_path = tmp_path / "model.joblib"
        
        save_model(mock_model, model_path)
        loaded = load_model(model_path)
        
        assert loaded is not None
    
    def test_load_model_not_found(self, tmp_path):
        """Testa carregar modelo inexistente."""
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.joblib")
    
    def test_validate_dataframe_valid(self):
        """Testa validação de DataFrame válido."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        
        assert validate_dataframe(df, ["A", "B"]) is True
    
    def test_validate_dataframe_missing_columns(self):
        """Testa validação com colunas faltando."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        
        with pytest.raises(ValueError, match="Colunas obrigatórias faltando"):
            validate_dataframe(df, ["A", "B", "C"])
    
    def test_validate_dataframe_no_raise(self):
        """Testa validação sem levantar erro."""
        df = pd.DataFrame({"A": [1, 2]})
        
        result = validate_dataframe(df, ["A", "B"], raise_error=False)
        
        assert result is False
    
    def test_clean_column_names(self):
        """Testa limpeza de nomes de colunas."""
        df = pd.DataFrame({
            "  Name  ": [1],
            "Value-Col": [2],
            "data.field": [3],
            "(extra)": [4]
        })
        
        df_clean = clean_column_names(df)
        
        assert "NAME" in df_clean.columns
        assert "VALUE_COL" in df_clean.columns
        assert "DATA_FIELD" in df_clean.columns
        assert "EXTRA" in df_clean.columns
    
    def test_categorize_inde(self):
        """Testa categorização de INDE."""
        assert categorize_inde(4.0) == "Quartzo"
        assert categorize_inde(6.0) == "Ágata"
        assert categorize_inde(7.5) == "Ametista"
        assert categorize_inde(9.0) == "Topázio"
    
    def test_categorize_inde_nan(self):
        """Testa categorização de INDE com NaN."""
        assert categorize_inde(np.nan) == "Desconhecido"
    
    def test_calculate_defasagem(self):
        """Testa cálculo de defasagem."""
        assert calculate_defasagem(fase_atual=5, fase_ideal=7) == 2
        assert calculate_defasagem(fase_atual=7, fase_ideal=5) == -2
        assert calculate_defasagem(fase_atual=5, fase_ideal=5) == 0
    
    def test_get_risk_level(self):
        """Testa obtenção de nível de risco."""
        assert get_risk_level(0) == "BAIXO"
        assert get_risk_level(1) == "MÉDIO"
        assert get_risk_level(2) == "ALTO"
        assert get_risk_level(99) == "DESCONHECIDO"
    
    def test_safe_divide_normal(self):
        """Testa divisão normal."""
        assert safe_divide(10, 2) == 5
    
    def test_safe_divide_by_zero(self):
        """Testa divisão por zero."""
        assert safe_divide(10, 0) == 0
        assert safe_divide(10, 0, fill_value=999) == 999
    
    def test_safe_divide_array(self):
        """Testa divisão de arrays."""
        num = np.array([10, 20, 30])
        denom = np.array([2, 0, 5])
        
        result = safe_divide(num, denom)
        
        np.testing.assert_array_equal(result, [5, 0, 6])
    
    def test_safe_divide_series(self):
        """Testa divisão de Series."""
        num = pd.Series([10, 20, 30])
        denom = pd.Series([2, 0, 5])
        
        result = safe_divide(num, denom)
        
        assert isinstance(result, pd.Series)
        assert result.iloc[1] == 0
    
    def test_get_memory_usage(self):
        """Testa obtenção de uso de memória."""
        df = pd.DataFrame({"A": range(1000)})
        
        usage = get_memory_usage(df)
        
        assert isinstance(usage, str)
        assert "KB" in usage or "bytes" in usage


class TestLogger:
    """Testes para funções de logging."""
    
    def test_setup_logger(self, tmp_path):
        """Testa configuração de logger."""
        log_file = tmp_path / "test.log"
        
        logger = setup_logger("test_logger", log_file=str(log_file))
        
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_setup_logger_without_file(self):
        """Testa logger sem arquivo."""
        logger = setup_logger("test_no_file", log_file=None)
        
        assert logger is not None
    
    def test_get_logger_existing(self, tmp_path):
        """Testa obtenção de logger existente."""
        log_file = tmp_path / "test.log"
        setup_logger("existing_logger", log_file=str(log_file))
        
        logger = get_logger("existing_logger")
        
        assert logger.name == "existing_logger"
    
    def test_get_logger_new(self):
        """Testa obtenção de novo logger."""
        logger = get_logger("brand_new_logger")
        
        assert logger is not None
    
    def test_logger_context_success(self, tmp_path):
        """Testa LoggerContext com sucesso."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("context_test", log_file=str(log_file))
        
        with LoggerContext(logger, "test operation"):
            pass  # Operação bem-sucedida
        
        # Não deve levantar exceção
    
    def test_logger_context_exception(self, tmp_path):
        """Testa LoggerContext com exceção."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("context_exc", log_file=str(log_file))
        
        with pytest.raises(ValueError):
            with LoggerContext(logger, "failing operation"):
                raise ValueError("Test error")
