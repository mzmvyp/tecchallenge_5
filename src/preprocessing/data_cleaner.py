"""
Módulo para limpeza de dados do projeto Passos Mágicos.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    INDICADOR_MIN,
    INDICADOR_MAX,
    COLUNAS_NUMERICAS,
    COLUNAS_CATEGORICAS,
    COLUNAS_BOOLEANAS,
)
from src.utils.logger import get_logger
from src.utils.helpers import validate_dataframe

logger = get_logger(__name__)


class DataCleaner:
    """Classe para limpeza e tratamento de dados."""
    
    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        remove_duplicates: bool = True
    ):
        """
        Inicializa o DataCleaner.
        
        Args:
            numeric_strategy: Estratégia para valores faltantes numéricos
                             ("median", "mean", "drop")
            categorical_strategy: Estratégia para valores faltantes categóricos
                                 ("mode", "unknown", "drop")
            remove_duplicates: Se deve remover duplicatas
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.remove_duplicates = remove_duplicates
        
        self._fill_values: Dict[str, any] = {}
        self._cleaning_report: Dict[str, any] = {}
        
        logger.info(
            f"DataCleaner inicializado: "
            f"numeric_strategy={numeric_strategy}, "
            f"categorical_strategy={categorical_strategy}"
        )
    
    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """
        Aprende parâmetros de limpeza do DataFrame.
        
        Args:
            df: DataFrame de treino
        
        Returns:
            Self para encadeamento
        """
        logger.info("Ajustando DataCleaner aos dados...")
        
        self._fill_values = {}
        
        # Calcular valores de preenchimento para numéricas
        for col in df.select_dtypes(include=[np.number]).columns:
            if self.numeric_strategy == "median":
                self._fill_values[col] = df[col].median()
            elif self.numeric_strategy == "mean":
                self._fill_values[col] = df[col].mean()
        
        # Calcular valores de preenchimento para categóricas
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if self.categorical_strategy == "mode":
                mode_val = df[col].mode()
                self._fill_values[col] = mode_val.iloc[0] if len(mode_val) > 0 else "DESCONHECIDO"
            elif self.categorical_strategy == "unknown":
                self._fill_values[col] = "DESCONHECIDO"
        
        logger.info(f"Valores de preenchimento calculados para {len(self._fill_values)} colunas")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformações de limpeza ao DataFrame.
        
        Args:
            df: DataFrame a ser limpo
        
        Returns:
            DataFrame limpo
        """
        logger.info(f"Iniciando limpeza de {len(df)} registros...")
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        # Inicializar relatório
        self._cleaning_report = {
            "registros_iniciais": initial_rows,
            "duplicatas_removidas": 0,
            "valores_faltantes_preenchidos": {},
            "valores_fora_range_corrigidos": 0,
        }
        
        # 1. Remover duplicatas (se configurado)
        if self.remove_duplicates:
            df_clean = self._remove_duplicates(df_clean)
        
        # 2. Tratar valores faltantes
        df_clean = self._handle_missing_values(df_clean)
        
        # 3. Validar ranges dos indicadores
        df_clean = self._validate_indicator_ranges(df_clean)
        
        # 4. Padronizar tipos de dados
        df_clean = self._standardize_types(df_clean)
        
        # Atualizar relatório
        self._cleaning_report["registros_finais"] = len(df_clean)
        self._cleaning_report["registros_removidos"] = initial_rows - len(df_clean)
        
        logger.info(
            f"Limpeza concluída: {initial_rows} -> {len(df_clean)} registros "
            f"({initial_rows - len(df_clean)} removidos)"
        )
        
        return df_clean
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajusta e transforma o DataFrame.
        
        Args:
            df: DataFrame a ser processado
        
        Returns:
            DataFrame limpo
        """
        return self.fit(df).transform(df)
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove registros duplicados."""
        initial_count = len(df)
        
        # Tentar usar RA como identificador único
        if "RA" in df.columns:
            # Manter o registro mais recente (maior ANO_PEDE)
            if "ANO_PEDE" in df.columns:
                df = df.sort_values("ANO_PEDE", ascending=False)
            df = df.drop_duplicates(subset=["RA"], keep="first")
        else:
            df = df.drop_duplicates()
        
        removed = initial_count - len(df)
        self._cleaning_report["duplicatas_removidas"] = removed
        
        if removed > 0:
            logger.info(f"Removidas {removed} duplicatas")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata valores faltantes."""
        missing_report = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            
            if missing_count == 0:
                continue
            
            missing_report[col] = missing_count
            
            if col in self._fill_values:
                df[col] = df[col].fillna(self._fill_values[col])
                logger.debug(f"Coluna '{col}': {missing_count} valores preenchidos")
        
        self._cleaning_report["valores_faltantes_preenchidos"] = missing_report
        
        return df
    
    def _validate_indicator_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e corrige ranges dos indicadores (0-10)."""
        # Usar apenas colunas numéricas específicas dos indicadores
        indicator_cols = [
            col for col in df.columns 
            if col in ["INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV",
                       "NOTA_PORT", "NOTA_MAT", "NOTA_ING"]
        ]
        
        corrections = 0
        
        for col in indicator_cols:
            if col not in df.columns:
                continue
            
            # Verificar se a coluna é numérica
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Tentar converter para numérico
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    logger.warning(f"Coluna '{col}' não é numérica e não pode ser convertida")
                    continue
            
            # Valores abaixo do mínimo
            below_min = df[col] < INDICADOR_MIN
            if below_min.any():
                df.loc[below_min, col] = INDICADOR_MIN
                corrections += below_min.sum()
            
            # Valores acima do máximo
            above_max = df[col] > INDICADOR_MAX
            if above_max.any():
                df.loc[above_max, col] = INDICADOR_MAX
                corrections += above_max.sum()
        
        self._cleaning_report["valores_fora_range_corrigidos"] = corrections
        
        if corrections > 0:
            logger.info(f"Corrigidos {corrections} valores fora do range [0-10]")
        
        return df
    
    def _standardize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza tipos de dados das colunas."""
        # Booleanos
        bool_cols = [col for col in COLUNAS_BOOLEANAS if col in df.columns]
        for col in bool_cols:
            df[col] = self._convert_to_boolean(df[col])
        
        # Categóricas
        cat_cols = [col for col in COLUNAS_CATEGORICAS if col in df.columns]
        for col in cat_cols:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        return df
    
    @staticmethod
    def _convert_to_boolean(series: pd.Series) -> pd.Series:
        """Converte série para booleano."""
        # Mapear valores comuns
        true_values = {"SIM", "S", "YES", "Y", "TRUE", "1", "VERDADEIRO", 1, True}
        false_values = {"NAO", "NÃO", "N", "NO", "FALSE", "0", "FALSO", 0, False}
        
        def convert_value(val):
            if pd.isna(val):
                return False
            
            val_str = str(val).strip().upper()
            
            if val in true_values or val_str in true_values:
                return True
            elif val in false_values or val_str in false_values:
                return False
            else:
                return False
        
        return series.apply(convert_value)
    
    def get_cleaning_report(self) -> Dict[str, any]:
        """
        Retorna relatório da última limpeza.
        
        Returns:
            Dicionário com estatísticas da limpeza
        """
        return self._cleaning_report.copy()
    
    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna resumo de valores faltantes.
        
        Args:
            df: DataFrame para analisar
        
        Returns:
            DataFrame com resumo de valores faltantes
        """
        missing = df.isnull().sum()
        missing_pct = df.isnull().mean() * 100
        
        summary = pd.DataFrame({
            "coluna": missing.index,
            "faltantes": missing.values,
            "percentual": missing_pct.values
        })
        
        return summary[summary["faltantes"] > 0].sort_values(
            "faltantes", ascending=False
        )


def clean_data(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode"
) -> pd.DataFrame:
    """
    Função de conveniência para limpar dados.
    
    Args:
        df: DataFrame a ser limpo
        numeric_strategy: Estratégia para numéricos
        categorical_strategy: Estratégia para categóricos
    
    Returns:
        DataFrame limpo
    """
    cleaner = DataCleaner(
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy
    )
    return cleaner.fit_transform(df)
