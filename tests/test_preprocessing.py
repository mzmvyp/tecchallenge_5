"""
Testes para o módulo de pré-processamento.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_cleaner import DataCleaner, clean_data
from src.preprocessing.data_transformer import DataTransformer, transform_data


class TestDataCleaner:
    """Testes para a classe DataCleaner."""
    
    def test_init_default_params(self):
        """Testa inicialização com parâmetros padrão."""
        cleaner = DataCleaner()
        
        assert cleaner.numeric_strategy == "median"
        assert cleaner.categorical_strategy == "mode"
        assert cleaner.remove_duplicates is True
    
    def test_init_custom_params(self):
        """Testa inicialização com parâmetros customizados."""
        cleaner = DataCleaner(
            numeric_strategy="mean",
            categorical_strategy="unknown",
            remove_duplicates=False
        )
        
        assert cleaner.numeric_strategy == "mean"
        assert cleaner.categorical_strategy == "unknown"
        assert cleaner.remove_duplicates is False
    
    def test_fit_calculates_fill_values(self, sample_dataframe):
        """Testa se fit calcula valores de preenchimento."""
        cleaner = DataCleaner()
        cleaner.fit(sample_dataframe)
        
        assert len(cleaner._fill_values) > 0
        assert "INDE" in cleaner._fill_values
    
    def test_transform_handles_missing_values(self, sample_dataframe_with_missing):
        """Testa tratamento de valores faltantes."""
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(sample_dataframe_with_missing)
        
        # Verificar que não há mais valores nulos nas colunas tratadas
        assert df_clean["INDE"].isnull().sum() == 0
        assert df_clean["IAN"].isnull().sum() == 0
    
    def test_transform_removes_duplicates(self, sample_dataframe):
        """Testa remoção de duplicatas."""
        # Criar duplicatas
        df_with_dups = pd.concat([sample_dataframe, sample_dataframe.iloc[:10]])
        
        cleaner = DataCleaner(remove_duplicates=True)
        df_clean = cleaner.fit_transform(df_with_dups)
        
        # Se RA existe, não deve haver duplicatas
        if "RA" in df_clean.columns:
            assert df_clean["RA"].duplicated().sum() == 0
    
    def test_transform_validates_indicator_ranges(self, sample_dataframe):
        """Testa validação de ranges dos indicadores."""
        df = sample_dataframe.copy()
        df.loc[0, "INDE"] = 15  # Valor fora do range
        df.loc[1, "IAN"] = -2  # Valor negativo
        
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(df)
        
        # Valores devem estar entre 0 e 10
        assert df_clean["INDE"].max() <= 10
        assert df_clean["IAN"].min() >= 0
    
    def test_get_cleaning_report(self, sample_dataframe_with_missing):
        """Testa geração de relatório de limpeza."""
        cleaner = DataCleaner()
        cleaner.fit_transform(sample_dataframe_with_missing)
        
        report = cleaner.get_cleaning_report()
        
        assert "registros_iniciais" in report
        assert "registros_finais" in report
        assert "duplicatas_removidas" in report
    
    def test_clean_data_convenience_function(self, sample_dataframe_with_missing):
        """Testa função de conveniência clean_data."""
        df_clean = clean_data(sample_dataframe_with_missing)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert len(df_clean) > 0


class TestDataTransformer:
    """Testes para a classe DataTransformer."""
    
    def test_init_default_params(self):
        """Testa inicialização com parâmetros padrão."""
        transformer = DataTransformer()
        
        assert transformer.scaling_method == "standard"
        assert transformer.encoding_method == "onehot"
    
    def test_fit_identifies_columns(self, sample_dataframe):
        """Testa se fit identifica colunas corretamente."""
        transformer = DataTransformer()
        transformer.fit(sample_dataframe)
        
        assert len(transformer._numeric_columns) > 0
        assert len(transformer._categorical_columns) >= 0
        assert transformer._fitted is True
    
    def test_transform_scales_numeric(self, sample_features_df):
        """Testa scaling de colunas numéricas."""
        transformer = DataTransformer(scaling_method="standard")
        df_transformed = transformer.fit_transform(sample_features_df)
        
        # Verificar que valores foram transformados
        assert df_transformed["INDE"].mean() != sample_features_df["INDE"].mean()
    
    def test_transform_without_fit_raises_error(self, sample_dataframe):
        """Testa que transform sem fit levanta erro."""
        transformer = DataTransformer()
        
        with pytest.raises(ValueError, match="não foi ajustado"):
            transformer.transform(sample_dataframe)
    
    def test_transform_with_onehot_encoding(self, sample_dataframe):
        """Testa one-hot encoding."""
        transformer = DataTransformer(encoding_method="onehot")
        df_transformed = transformer.fit_transform(sample_dataframe)
        
        # Deve ter mais colunas após one-hot
        assert len(df_transformed.columns) >= len(sample_dataframe.columns)
    
    def test_transform_with_label_encoding(self, sample_dataframe):
        """Testa label encoding."""
        transformer = DataTransformer(encoding_method="label")
        df_transformed = transformer.fit_transform(sample_dataframe)
        
        # Colunas categóricas devem ser numéricas
        if "GENERO" in df_transformed.columns:
            assert df_transformed["GENERO"].dtype in [np.int64, np.int32, np.float64]
    
    def test_transform_with_no_scaling(self, sample_features_df):
        """Testa sem scaling."""
        transformer = DataTransformer(scaling_method="none")
        df_transformed = transformer.fit_transform(sample_features_df)
        
        # Valores originais devem ser mantidos
        np.testing.assert_array_almost_equal(
            df_transformed["INDE"].values,
            sample_features_df["INDE"].values
        )
    
    def test_get_feature_names(self, sample_dataframe):
        """Testa obtenção de nomes de features."""
        transformer = DataTransformer(encoding_method="onehot")
        transformer.fit(sample_dataframe)
        
        feature_names = transformer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_transform_data_convenience_function(self, sample_dataframe):
        """Testa função de conveniência transform_data."""
        df_transformed, transformer = transform_data(sample_dataframe)
        
        assert isinstance(df_transformed, pd.DataFrame)
        assert isinstance(transformer, DataTransformer)
