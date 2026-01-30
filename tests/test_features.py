"""
Testes para o módulo de features.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    FeatureEngineer,
    calculate_risco_defasagem,
    engineer_features,
)
from src.features.feature_selection import FeatureSelector, select_features


class TestFeatureEngineer:
    """Testes para a classe FeatureEngineer."""
    
    def test_init(self):
        """Testa inicialização."""
        engineer = FeatureEngineer()
        assert len(engineer._created_features) == 0
    
    def test_create_defasagem_features(self, sample_dataframe):
        """Testa criação de features de defasagem."""
        engineer = FeatureEngineer()
        df_result = engineer.create_defasagem_features(sample_dataframe)
        
        if "IDADE" in sample_dataframe.columns:
            assert "FASE_IDEAL" in df_result.columns
        
        if "FASE" in sample_dataframe.columns and "FASE_IDEAL" in df_result.columns:
            assert "DEFASAGEM" in df_result.columns
            assert "DEFASAGEM_ABS" in df_result.columns
            assert "TEM_DEFASAGEM" in df_result.columns
    
    def test_create_temporal_features(self, sample_dataframe):
        """Testa criação de features temporais."""
        df = sample_dataframe.copy()
        df["ANO_INGRESSO"] = 2022
        
        engineer = FeatureEngineer()
        df_result = engineer.create_temporal_features(df)
        
        if "ANO_INGRESSO" in df.columns and "ANO_PEDE" in df.columns:
            assert "ANOS_PM" in df_result.columns
            assert df_result["ANOS_PM"].min() >= 0
    
    def test_create_indicator_features(self, sample_dataframe):
        """Testa criação de features de indicadores."""
        engineer = FeatureEngineer()
        df_result = engineer.create_indicator_features(sample_dataframe)
        
        # Deve ter criado métricas agregadas
        assert "MEDIA_INDICADORES" in df_result.columns
        assert "STD_INDICADORES" in df_result.columns
        assert "MIN_INDICADOR" in df_result.columns
        assert "MAX_INDICADOR" in df_result.columns
        assert "INDICADORES_BAIXOS" in df_result.columns
    
    def test_create_interaction_features(self, sample_dataframe):
        """Testa criação de features de interação."""
        engineer = FeatureEngineer()
        df_result = engineer.create_interaction_features(sample_dataframe)
        
        if "IDA" in sample_dataframe.columns and "IEG" in sample_dataframe.columns:
            assert "RATIO_IDA_IEG" in df_result.columns
            assert "ENGAJ_X_DESEMP" in df_result.columns
    
    def test_create_target(self, sample_dataframe):
        """Testa criação do target."""
        engineer = FeatureEngineer()
        
        # Primeiro criar features necessárias
        df = engineer.create_defasagem_features(sample_dataframe)
        df = engineer.create_indicator_features(df)
        df = engineer.create_target(df)
        
        assert "RISCO_DEFASAGEM" in df.columns
        assert "NIVEL_RISCO" in df.columns
        
        # Valores válidos
        assert set(df["RISCO_DEFASAGEM"].unique()).issubset({0, 1, 2})
        assert set(df["NIVEL_RISCO"].unique()).issubset({"BAIXO", "MÉDIO", "ALTO"})
    
    def test_create_all_features(self, sample_dataframe):
        """Testa criação de todas as features."""
        engineer = FeatureEngineer()
        df_result = engineer.create_all_features(sample_dataframe)
        
        # Deve ter mais colunas
        assert len(df_result.columns) > len(sample_dataframe.columns)
        
        # Deve ter criado features
        created = engineer.get_created_features()
        assert len(created) > 0
    
    def test_engineer_features_convenience(self, sample_dataframe):
        """Testa função de conveniência."""
        df_result = engineer_features(sample_dataframe)
        
        assert isinstance(df_result, pd.DataFrame)
        assert "RISCO_DEFASAGEM" in df_result.columns


class TestCalculateRiscoDefasagem:
    """Testes para função calculate_risco_defasagem."""
    
    def test_alto_risco_defasagem_alta(self):
        """Testa alto risco por defasagem alta."""
        row = pd.Series({"DEFASAGEM": 3, "INDE": 7.0, "INDICADORES_BAIXOS": 0})
        assert calculate_risco_defasagem(row) == 2
    
    def test_alto_risco_inde_baixo(self):
        """Testa alto risco por INDE baixo."""
        row = pd.Series({"DEFASAGEM": 0, "INDE": 4.0, "INDICADORES_BAIXOS": 0})
        assert calculate_risco_defasagem(row) == 2
    
    def test_alto_risco_muitos_indicadores_baixos(self):
        """Testa alto risco por muitos indicadores baixos."""
        row = pd.Series({"DEFASAGEM": 0, "INDE": 7.0, "INDICADORES_BAIXOS": 5})
        assert calculate_risco_defasagem(row) == 2
    
    def test_medio_risco_defasagem_um(self):
        """Testa médio risco por defasagem de 1."""
        row = pd.Series({"DEFASAGEM": 1, "INDE": 7.0, "INDICADORES_BAIXOS": 0})
        assert calculate_risco_defasagem(row) == 1
    
    def test_medio_risco_inde_medio(self):
        """Testa médio risco por INDE médio."""
        row = pd.Series({"DEFASAGEM": 0, "INDE": 6.0, "INDICADORES_BAIXOS": 0})
        assert calculate_risco_defasagem(row) == 1
    
    def test_baixo_risco(self):
        """Testa baixo risco."""
        row = pd.Series({"DEFASAGEM": 0, "INDE": 8.0, "INDICADORES_BAIXOS": 0})
        assert calculate_risco_defasagem(row) == 0


class TestFeatureSelector:
    """Testes para a classe FeatureSelector."""
    
    def test_init_default_params(self):
        """Testa inicialização com parâmetros padrão."""
        selector = FeatureSelector()
        
        assert selector.method == "importance"
        assert selector.variance_threshold == 0.01
        assert selector.correlation_threshold == 0.95
    
    def test_fit_selects_features(self, sample_features_df, sample_target):
        """Testa que fit seleciona features."""
        selector = FeatureSelector(method="importance", n_features=5)
        selector.fit(sample_features_df, sample_target)
        
        selected = selector.get_selected_features()
        
        assert len(selected) == 5
        assert all(f in sample_features_df.columns for f in selected)
    
    def test_transform_returns_selected_features(self, sample_features_df, sample_target):
        """Testa que transform retorna apenas features selecionadas."""
        selector = FeatureSelector(n_features=5)
        selector.fit(sample_features_df, sample_target)
        
        df_selected = selector.transform(sample_features_df)
        
        assert len(df_selected.columns) == 5
    
    def test_fit_transform(self, sample_features_df, sample_target):
        """Testa fit_transform."""
        selector = FeatureSelector(n_features=5)
        df_selected = selector.fit_transform(sample_features_df, sample_target)
        
        assert len(df_selected.columns) == 5
    
    def test_removes_low_variance_features(self, sample_features_df, sample_target):
        """Testa remoção de features com baixa variância."""
        # Adicionar coluna constante
        df = sample_features_df.copy()
        df["CONSTANT"] = 1
        
        selector = FeatureSelector(variance_threshold=0.01)
        selector.fit(df, sample_target)
        
        removed = selector.get_removed_features()
        assert "CONSTANT" in removed
    
    def test_removes_correlated_features(self, sample_features_df, sample_target):
        """Testa remoção de features correlacionadas."""
        # Adicionar coluna altamente correlacionada
        df = sample_features_df.copy()
        df["INDE_COPY"] = df["INDE"] * 1.001  # Muito correlacionada
        
        selector = FeatureSelector(correlation_threshold=0.99)
        selector.fit(df, sample_target)
        
        # Uma das duas deve ser removida
        selected = selector.get_selected_features()
        assert not ("INDE" in selected and "INDE_COPY" in selected)
    
    def test_get_feature_importances(self, sample_features_df, sample_target):
        """Testa obtenção de importâncias."""
        selector = FeatureSelector(method="importance")
        selector.fit(sample_features_df, sample_target)
        
        importances = selector.get_feature_importances()
        
        assert len(importances) > 0
        assert all(v >= 0 for v in importances.values())
    
    def test_select_features_convenience(self, sample_features_df, sample_target):
        """Testa função de conveniência select_features."""
        df_selected, selector = select_features(
            sample_features_df, 
            sample_target,
            n_features=5
        )
        
        assert len(df_selected.columns) == 5
        assert isinstance(selector, FeatureSelector)
    
    def test_selection_report(self, sample_features_df, sample_target):
        """Testa geração de relatório de seleção."""
        selector = FeatureSelector(n_features=5)
        selector.fit(sample_features_df, sample_target)
        
        report = selector.get_selection_report()
        
        assert "method" in report
        assert "n_selected" in report
        assert "selected_features" in report
