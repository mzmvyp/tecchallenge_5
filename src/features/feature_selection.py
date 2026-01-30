"""
Módulo de seleção de features para o projeto Passos Mágicos.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)

from src.config import FEATURES_MODELO
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Classe para seleção de features."""
    
    def __init__(
        self,
        method: str = "importance",
        n_features: Optional[int] = None,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95
    ):
        """
        Inicializa o FeatureSelector.
        
        Args:
            method: Método de seleção ("importance", "rfe", "kbest", "all")
            n_features: Número de features a selecionar (None = automático)
            variance_threshold: Threshold mínimo de variância
            correlation_threshold: Threshold para remoção de multicolinearidade
        """
        self.method = method
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        
        self._selected_features: List[str] = []
        self._feature_importances: Dict[str, float] = {}
        self._removed_features: Dict[str, str] = {}  # feature: motivo
        
        logger.info(
            f"FeatureSelector inicializado: method={method}, "
            f"variance_threshold={variance_threshold}"
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: Optional[List[str]] = None
    ) -> "FeatureSelector":
        """
        Ajusta o seletor às features.
        
        Args:
            X: Features
            y: Target
            categorical_features: Lista de features categóricas (para não aplicar variance)
        
        Returns:
            Self para encadeamento
        """
        logger.info(f"Ajustando FeatureSelector com {len(X.columns)} features...")
        
        # Identificar features categóricas
        if categorical_features is None:
            categorical_features = list(
                X.select_dtypes(include=["object", "category"]).columns
            )
        
        # Manter apenas features numéricas para seleção
        numeric_features = [
            col for col in X.columns 
            if col not in categorical_features
        ]
        
        X_numeric = X[numeric_features].copy()
        
        # 1. Remover features com baixa variância
        X_numeric = self._remove_low_variance(X_numeric)
        
        # 2. Remover features altamente correlacionadas
        X_numeric = self._remove_high_correlation(X_numeric)
        
        # 3. Selecionar features pelo método escolhido
        remaining_features = list(X_numeric.columns)
        
        if self.method == "importance":
            self._select_by_importance(X_numeric, y)
        elif self.method == "rfe":
            self._select_by_rfe(X_numeric, y)
        elif self.method == "kbest":
            self._select_by_kbest(X_numeric, y)
        elif self.method == "all":
            self._selected_features = remaining_features
        
        # Adicionar features categóricas de volta
        self._selected_features.extend(categorical_features)
        
        logger.info(f"Features selecionadas: {len(self._selected_features)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Seleciona apenas as features escolhidas.
        
        Args:
            X: DataFrame com todas as features
        
        Returns:
            DataFrame com features selecionadas
        """
        if not self._selected_features:
            raise ValueError("Seletor não foi ajustado. Chame fit() primeiro.")
        
        # Filtrar features que existem no DataFrame
        available_features = [
            col for col in self._selected_features 
            if col in X.columns
        ]
        
        return X[available_features]
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ajusta e transforma em uma única operação.
        
        Args:
            X: Features
            y: Target
            categorical_features: Features categóricas
        
        Returns:
            DataFrame com features selecionadas
        """
        return self.fit(X, y, categorical_features).transform(X)
    
    def _remove_low_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features com baixa variância."""
        if self.variance_threshold <= 0:
            return X
        
        # Calcular variância normalizada
        X_filled = X.fillna(X.median())
        
        # Normalizar para comparar variâncias
        X_normalized = (X_filled - X_filled.min()) / (X_filled.max() - X_filled.min() + 1e-10)
        
        variances = X_normalized.var()
        low_variance_cols = variances[variances < self.variance_threshold].index.tolist()
        
        for col in low_variance_cols:
            self._removed_features[col] = f"Baixa variância ({variances[col]:.4f})"
        
        if low_variance_cols:
            logger.info(f"Removidas {len(low_variance_cols)} features com baixa variância")
        
        return X.drop(columns=low_variance_cols, errors="ignore")
    
    def _remove_high_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features altamente correlacionadas (multicolinearidade)."""
        if self.correlation_threshold >= 1.0:
            return X
        
        X_filled = X.fillna(X.median())
        
        # Calcular matriz de correlação
        corr_matrix = X_filled.corr().abs()
        
        # Identificar pares correlacionados
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Encontrar features a remover
        cols_to_drop = []
        
        for col in upper_triangle.columns:
            high_corr = upper_triangle[col][upper_triangle[col] > self.correlation_threshold]
            
            if not high_corr.empty:
                # Manter a feature com maior variância
                variances = X_filled[[col] + high_corr.index.tolist()].var()
                
                for corr_col in high_corr.index:
                    if variances[col] >= variances[corr_col]:
                        if corr_col not in cols_to_drop:
                            cols_to_drop.append(corr_col)
                            self._removed_features[corr_col] = \
                                f"Alta correlação com {col} ({high_corr[corr_col]:.3f})"
        
        if cols_to_drop:
            logger.info(f"Removidas {len(cols_to_drop)} features por alta correlação")
        
        return X.drop(columns=cols_to_drop, errors="ignore")
    
    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Seleciona features por importância (Random Forest)."""
        logger.info("Selecionando features por importância (Random Forest)...")
        
        X_filled = X.fillna(X.median())
        
        # Treinar Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        rf.fit(X_filled, y)
        
        # Obter importâncias
        importances = pd.Series(
            rf.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        self._feature_importances = importances.to_dict()
        
        # Selecionar top N features
        n_features = self.n_features or max(10, len(X.columns) // 2)
        self._selected_features = importances.head(n_features).index.tolist()
        
        logger.debug(f"Top 10 features por importância: {importances.head(10).to_dict()}")
    
    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Seleciona features por RFE (Recursive Feature Elimination)."""
        logger.info("Selecionando features por RFE...")
        
        X_filled = X.fillna(X.median())
        
        # Estimador base
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        
        # RFE
        n_features = self.n_features or max(10, len(X.columns) // 2)
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=1
        )
        rfe.fit(X_filled, y)
        
        # Features selecionadas
        self._selected_features = X.columns[rfe.support_].tolist()
        
        # Ranking
        ranking = pd.Series(rfe.ranking_, index=X.columns)
        self._feature_importances = (1 / ranking).to_dict()
    
    def _select_by_kbest(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Seleciona features por SelectKBest."""
        logger.info("Selecionando features por SelectKBest (f_classif)...")
        
        X_filled = X.fillna(X.median())
        
        # SelectKBest
        n_features = self.n_features or max(10, len(X.columns) // 2)
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X_filled, y)
        
        # Features selecionadas
        self._selected_features = X.columns[selector.get_support()].tolist()
        
        # Scores
        scores = pd.Series(selector.scores_, index=X.columns)
        self._feature_importances = scores.to_dict()
    
    def get_selected_features(self) -> List[str]:
        """
        Retorna lista de features selecionadas.
        
        Returns:
            Lista de nomes de features
        """
        return self._selected_features.copy()
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Retorna importâncias das features.
        
        Returns:
            Dicionário {feature: importância}
        """
        return self._feature_importances.copy()
    
    def get_removed_features(self) -> Dict[str, str]:
        """
        Retorna features removidas e motivos.
        
        Returns:
            Dicionário {feature: motivo}
        """
        return self._removed_features.copy()
    
    def get_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna matriz de correlação das features.
        
        Args:
            X: DataFrame com features
        
        Returns:
            Matriz de correlação
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        return X[numeric_cols].corr()
    
    def get_selection_report(self) -> Dict:
        """
        Gera relatório da seleção de features.
        
        Returns:
            Dicionário com informações da seleção
        """
        return {
            "method": self.method,
            "n_selected": len(self._selected_features),
            "selected_features": self._selected_features,
            "n_removed": len(self._removed_features),
            "removed_features": self._removed_features,
            "top_10_importances": dict(
                sorted(
                    self._feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )
        }


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "importance",
    n_features: Optional[int] = None
) -> Tuple[pd.DataFrame, FeatureSelector]:
    """
    Função de conveniência para seleção de features.
    
    Args:
        X: DataFrame com features
        y: Target
        method: Método de seleção
        n_features: Número de features a selecionar
    
    Returns:
        Tupla (DataFrame com features selecionadas, Seletor ajustado)
    """
    selector = FeatureSelector(method=method, n_features=n_features)
    X_selected = selector.fit_transform(X, y)
    
    return X_selected, selector
