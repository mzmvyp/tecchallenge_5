"""
Módulo para transformação de dados do projeto Passos Mágicos.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

from src.config import (
    COLUNAS_NUMERICAS,
    COLUNAS_CATEGORICAS,
    PEDRA_LIMITES,
)
from src.utils.logger import get_logger
from src.utils.helpers import categorize_inde

logger = get_logger(__name__)


class DataTransformer:
    """Classe para transformação e encoding de dados."""
    
    def __init__(
        self,
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
        handle_unknown: str = "ignore"
    ):
        """
        Inicializa o DataTransformer.
        
        Args:
            scaling_method: Método de scaling ("standard", "minmax", "none")
            encoding_method: Método de encoding ("onehot", "label", "none")
            handle_unknown: Como lidar com categorias desconhecidas ("ignore", "error")
        """
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.handle_unknown = handle_unknown
        
        self._scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self._encoders: Dict[str, Union[LabelEncoder, OneHotEncoder]] = {}
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._fitted = False
        
        logger.info(
            f"DataTransformer inicializado: "
            f"scaling={scaling_method}, encoding={encoding_method}"
        )
    
    def fit(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> "DataTransformer":
        """
        Ajusta os transformadores aos dados.
        
        Args:
            df: DataFrame de treino
            numeric_columns: Colunas numéricas a transformar
            categorical_columns: Colunas categóricas a transformar
        
        Returns:
            Self para encadeamento
        """
        logger.info("Ajustando DataTransformer aos dados...")
        
        # Identificar colunas
        if numeric_columns is None:
            self._numeric_columns = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in ["ANO_PEDE", "RA"]
            ]
        else:
            self._numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        if categorical_columns is None:
            self._categorical_columns = [
                col for col in df.select_dtypes(include=["object", "category"]).columns
            ]
        else:
            self._categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        # Ajustar scaler para colunas numéricas
        if self.scaling_method != "none" and self._numeric_columns:
            self._fit_scaler(df[self._numeric_columns])
        
        # Ajustar encoders para colunas categóricas
        if self.encoding_method != "none" and self._categorical_columns:
            self._fit_encoders(df[self._categorical_columns])
        
        self._fitted = True
        
        logger.info(
            f"Transformer ajustado: {len(self._numeric_columns)} numéricas, "
            f"{len(self._categorical_columns)} categóricas"
        )
        
        return self
    
    def _fit_scaler(self, df_numeric: pd.DataFrame) -> None:
        """Ajusta o scaler."""
        if self.scaling_method == "standard":
            self._scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self._scaler = MinMaxScaler()
        
        if self._scaler is not None:
            # Tratar NaN antes de ajustar
            df_filled = df_numeric.fillna(df_numeric.median())
            self._scaler.fit(df_filled)
    
    def _fit_encoders(self, df_categorical: pd.DataFrame) -> None:
        """Ajusta os encoders."""
        for col in df_categorical.columns:
            if self.encoding_method == "label":
                encoder = LabelEncoder()
                encoder.fit(df_categorical[col].astype(str).fillna("DESCONHECIDO"))
            elif self.encoding_method == "onehot":
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown=self.handle_unknown
                )
                encoder.fit(df_categorical[[col]].astype(str).fillna("DESCONHECIDO"))
            
            self._encoders[col] = encoder
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformações ao DataFrame.
        
        Args:
            df: DataFrame a ser transformado
        
        Returns:
            DataFrame transformado
        """
        if not self._fitted:
            raise ValueError("Transformer não foi ajustado. Chame fit() primeiro.")
        
        logger.info(f"Transformando {len(df)} registros...")
        
        df_transformed = df.copy()
        
        # Aplicar scaling
        if self._scaler is not None and self._numeric_columns:
            cols_to_scale = [col for col in self._numeric_columns if col in df.columns]
            if cols_to_scale:
                df_filled = df_transformed[cols_to_scale].fillna(
                    df_transformed[cols_to_scale].median()
                )
                df_transformed[cols_to_scale] = self._scaler.transform(df_filled)
        
        # Aplicar encoding
        if self.encoding_method == "onehot" and self._encoders:
            df_transformed = self._apply_onehot_encoding(df_transformed)
        elif self.encoding_method == "label" and self._encoders:
            df_transformed = self._apply_label_encoding(df_transformed)
        
        logger.info(f"Transformação concluída: {len(df_transformed.columns)} colunas")
        
        return df_transformed
    
    def _apply_onehot_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica one-hot encoding."""
        encoded_dfs = [df]
        cols_to_drop = []
        
        for col, encoder in self._encoders.items():
            if col not in df.columns:
                continue
            
            # Criar matriz one-hot
            encoded = encoder.transform(
                df[[col]].astype(str).fillna("DESCONHECIDO")
            )
            
            # Criar nomes de colunas
            feature_names = encoder.get_feature_names_out([col])
            encoded_df = pd.DataFrame(
                encoded,
                columns=feature_names,
                index=df.index
            )
            
            encoded_dfs.append(encoded_df)
            cols_to_drop.append(col)
        
        # Concatenar e remover colunas originais
        result = pd.concat(encoded_dfs, axis=1)
        result = result.drop(columns=cols_to_drop, errors="ignore")
        
        return result
    
    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica label encoding."""
        for col, encoder in self._encoders.items():
            if col not in df.columns:
                continue
            
            df[col] = encoder.transform(
                df[col].astype(str).fillna("DESCONHECIDO")
            )
        
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ajusta e transforma o DataFrame.
        
        Args:
            df: DataFrame a ser processado
            numeric_columns: Colunas numéricas
            categorical_columns: Colunas categóricas
        
        Returns:
            DataFrame transformado
        """
        return self.fit(df, numeric_columns, categorical_columns).transform(df)
    
    def inverse_transform_numeric(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Reverte transformação de colunas numéricas.
        
        Args:
            df: DataFrame transformado
            columns: Colunas a reverter (None = todas numéricas)
        
        Returns:
            DataFrame com valores originais
        """
        if self._scaler is None:
            return df
        
        df_result = df.copy()
        
        if columns is None:
            columns = self._numeric_columns
        
        cols_to_inverse = [col for col in columns if col in df.columns]
        if cols_to_inverse:
            df_result[cols_to_inverse] = self._scaler.inverse_transform(
                df[cols_to_inverse]
            )
        
        return df_result
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nomes de todas as features após transformação.
        
        Returns:
            Lista de nomes de features
        """
        feature_names = list(self._numeric_columns)
        
        if self.encoding_method == "onehot":
            for col, encoder in self._encoders.items():
                feature_names.extend(encoder.get_feature_names_out([col]))
        else:
            feature_names.extend(self._categorical_columns)
        
        return feature_names


def create_pedra_dummies(df: pd.DataFrame, pedra_column: str = "PEDRA") -> pd.DataFrame:
    """
    Cria dummies para classificação PEDRA.
    
    Args:
        df: DataFrame original
        pedra_column: Nome da coluna PEDRA
    
    Returns:
        DataFrame com dummies adicionadas
    """
    if pedra_column not in df.columns:
        logger.warning(f"Coluna '{pedra_column}' não encontrada")
        return df
    
    # Criar dummies
    pedra_dummies = pd.get_dummies(
        df[pedra_column],
        prefix="PEDRA",
        dtype=int
    )
    
    # Concatenar com DataFrame original
    df_result = pd.concat([df, pedra_dummies], axis=1)
    
    return df_result


def transform_data(
    df: pd.DataFrame,
    scaling_method: str = "standard",
    encoding_method: str = "onehot"
) -> Tuple[pd.DataFrame, DataTransformer]:
    """
    Função de conveniência para transformar dados.
    
    Args:
        df: DataFrame a ser transformado
        scaling_method: Método de scaling
        encoding_method: Método de encoding
    
    Returns:
        Tupla (DataFrame transformado, Transformer ajustado)
    """
    transformer = DataTransformer(
        scaling_method=scaling_method,
        encoding_method=encoding_method
    )
    
    df_transformed = transformer.fit_transform(df)
    
    return df_transformed, transformer
