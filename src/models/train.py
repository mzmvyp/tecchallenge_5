"""
Módulo de treinamento de modelos para o projeto Passos Mágicos.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    MODELS_DIR,
    MODEL_CONFIGS,
    TRAIN_CONFIG,
    settings,
)
from src.utils.logger import get_logger, LoggerContext
from src.utils.helpers import save_model, ensure_dir

logger = get_logger(__name__)

# Tentar importar XGBoost (opcional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost não disponível")

# Tentar importar SMOTE (opcional)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn não disponível")


class ModelTrainer:
    """Classe para treinamento de modelos de ML."""
    
    def __init__(
        self,
        model_type: str = "random_forest",
        use_smote: bool = True,
        cv_folds: int = 5,
        scoring: str = "recall_weighted",
        random_state: int = 42
    ):
        """
        Inicializa o ModelTrainer.
        
        Args:
            model_type: Tipo de modelo ("logistic_regression", "random_forest",
                       "gradient_boosting", "xgboost", "svm")
            use_smote: Se deve usar SMOTE para balanceamento
            cv_folds: Número de folds para validação cruzada
            scoring: Métrica para otimização
            random_state: Seed para reprodutibilidade
        """
        self.model_type = model_type
        self.use_smote = use_smote and IMBLEARN_AVAILABLE
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        
        self.model: Optional[Any] = None
        self.pipeline: Optional[Pipeline] = None
        self.best_params: Dict[str, Any] = {}
        self.cv_results: Dict[str, Any] = {}
        
        logger.info(
            f"ModelTrainer inicializado: model_type={model_type}, "
            f"use_smote={self.use_smote}, scoring={scoring}"
        )
    
    def _get_base_model(self) -> Any:
        """Retorna instância do modelo base."""
        models = {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            "random_forest": RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "svm": SVC(
                random_state=self.random_state,
                probability=True
            ),
        }
        
        if XGBOOST_AVAILABLE:
            models["xgboost"] = XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        
        if self.model_type not in models:
            raise ValueError(
                f"Tipo de modelo '{self.model_type}' não suportado. "
                f"Opções: {list(models.keys())}"
            )
        
        return models[self.model_type]
    
    def _get_param_grid(self) -> Dict[str, List]:
        """Retorna grid de hiperparâmetros para otimização."""
        # Usar configurações do config.py se disponíveis
        if self.model_type in MODEL_CONFIGS:
            param_grid = {}
            for key, values in MODEL_CONFIGS[self.model_type].items():
                # Adicionar prefixo do modelo no pipeline
                param_grid[f"model__{key}"] = values
            return param_grid
        
        # Configurações padrão
        default_grids = {
            "logistic_regression": {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": ["balanced", None],
            },
            "random_forest": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20, None],
                "model__class_weight": ["balanced"],
            },
            "gradient_boosting": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.01, 0.1],
            },
            "xgboost": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.01, 0.1],
            },
            "svm": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["rbf", "linear"],
                "model__class_weight": ["balanced"],
            },
        }
        
        return default_grids.get(self.model_type, {})
    
    def _build_pipeline(self) -> Pipeline:
        """Constrói pipeline de preprocessamento e modelo."""
        steps = []
        
        # Scaler
        steps.append(("scaler", StandardScaler()))
        
        # SMOTE (se disponível e configurado)
        if self.use_smote:
            steps.append(("smote", SMOTE(random_state=self.random_state)))
        
        # Modelo
        steps.append(("model", self._get_base_model()))
        
        # Usar ImbPipeline se SMOTE estiver incluído
        if self.use_smote:
            return ImbPipeline(steps)
        
        return Pipeline(steps)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        optimize_hyperparams: bool = True
    ) -> "ModelTrainer":
        """
        Treina o modelo.
        
        Args:
            X: Features de treino
            y: Target de treino
            optimize_hyperparams: Se deve otimizar hiperparâmetros
        
        Returns:
            Self para encadeamento
        """
        with LoggerContext(logger, f"Treinamento {self.model_type}"):
            # Construir pipeline
            self.pipeline = self._build_pipeline()
            
            # Converter para numpy se necessário
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            y_array = y.values if isinstance(y, pd.Series) else y
            
            if optimize_hyperparams:
                self._train_with_optimization(X_array, y_array)
            else:
                self._train_simple(X_array, y_array)
            
            # Armazenar modelo final
            self.model = self.pipeline.named_steps["model"]
            
            logger.info(f"Modelo treinado com sucesso")
            
        return self
    
    def _train_simple(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina modelo sem otimização de hiperparâmetros."""
        logger.info("Treinando modelo sem otimização de hiperparâmetros...")
        
        self.pipeline.fit(X, y)
        
        # Validação cruzada para métricas
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        cv_scores = cross_val_score(
            self.pipeline, X, y,
            cv=cv, scoring=self.scoring
        )
        
        self.cv_results = {
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "all_scores": cv_scores.tolist()
        }
        
        logger.info(
            f"CV {self.scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )
    
    def _train_with_optimization(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina modelo com otimização de hiperparâmetros."""
        logger.info("Treinando modelo com otimização de hiperparâmetros...")
        
        param_grid = self._get_param_grid()
        
        if not param_grid:
            logger.warning("Nenhum grid de parâmetros definido, usando treino simples")
            return self._train_simple(X, y)
        
        # Estratificação
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Grid Search
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        
        grid_search.fit(X, y)
        
        # Atualizar pipeline com melhor modelo
        self.pipeline = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Resultados
        self.cv_results = {
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "cv_results": {
                "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
            }
        }
        
        logger.info(f"Melhor {self.scoring}: {grid_search.best_score_:.4f}")
        logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
    
    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        include_metadata: bool = True,
        feature_names: Optional[List[str]] = None
    ) -> Path:
        """
        Salva o modelo treinado.
        
        Args:
            path: Caminho para salvar (None = padrão)
            include_metadata: Se deve incluir metadados
            feature_names: Nomes das features usadas
        
        Returns:
            Caminho do arquivo salvo
        """
        if self.pipeline is None:
            raise ValueError("Nenhum modelo treinado para salvar")
        
        if path is None:
            ensure_dir(MODELS_DIR)
            path = MODELS_DIR / "model.joblib"
        
        path = Path(path)
        
        # Objeto a salvar
        model_data = {
            "pipeline": self.pipeline,
            "model_type": self.model_type,
            "best_params": self.best_params,
            "cv_results": self.cv_results,
            "feature_names": feature_names or [],
        }
        
        if include_metadata:
            model_data["metadata"] = {
                "version": settings.model_version,
                "scoring": self.scoring,
                "use_smote": self.use_smote,
            }
        
        save_model(model_data, path)
        
        logger.info(f"Modelo salvo em: {path}")
        
        return path
    
    def get_feature_importances(
        self,
        feature_names: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Retorna importâncias das features (se disponível).
        
        Args:
            feature_names: Nomes das features
        
        Returns:
            Series com importâncias
        """
        if self.model is None:
            raise ValueError("Nenhum modelo treinado")
        
        # Verificar se modelo tem feature_importances_
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).flatten()
        else:
            logger.warning("Modelo não suporta importância de features")
            return pd.Series()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return pd.Series(
            importances,
            index=feature_names
        ).sort_values(ascending=False)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    optimize: bool = True,
    save_path: Optional[Path] = None
) -> Tuple[ModelTrainer, Path]:
    """
    Função de conveniência para treinar e salvar modelo.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        model_type: Tipo de modelo
        optimize: Se deve otimizar hiperparâmetros
        save_path: Caminho para salvar
    
    Returns:
        Tupla (Trainer, caminho do modelo salvo)
    """
    trainer = ModelTrainer(model_type=model_type)
    trainer.train(X_train, y_train, optimize_hyperparams=optimize)
    
    model_path = trainer.save(save_path)
    
    return trainer, model_path


def train_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_types: Optional[List[str]] = None
) -> Dict[str, ModelTrainer]:
    """
    Treina múltiplos modelos para comparação.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        model_types: Lista de tipos de modelo
    
    Returns:
        Dicionário {model_type: trainer}
    """
    if model_types is None:
        model_types = ["logistic_regression", "random_forest", "gradient_boosting"]
        if XGBOOST_AVAILABLE:
            model_types.append("xgboost")
    
    trainers = {}
    
    for model_type in model_types:
        logger.info(f"Treinando {model_type}...")
        try:
            trainer = ModelTrainer(model_type=model_type)
            trainer.train(X_train, y_train, optimize_hyperparams=True)
            trainers[model_type] = trainer
        except Exception as e:
            logger.error(f"Erro ao treinar {model_type}: {e}")
    
    return trainers


# Script principal para execução via CLI
if __name__ == "__main__":
    from src.preprocessing import DataLoader, DataCleaner, DataTransformer
    from src.features import FeatureEngineer, FeatureSelector
    
    logger.info("=== Iniciando pipeline de treinamento ===")
    
    # 1. Carregar dados
    logger.info("1. Carregando dados...")
    loader = DataLoader()
    df = loader.load_and_unify()
    
    # 2. Limpar dados
    logger.info("2. Limpando dados...")
    cleaner = DataCleaner()
    df = cleaner.fit_transform(df)
    
    # 3. Engenharia de features
    logger.info("3. Criando features...")
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    # 4. Preparar X e y
    target_col = "RISCO_DEFASAGEM"
    
    # IMPORTANTE: Para evitar DATA LEAKAGE, NÃO usar features que foram
    # diretamente usadas na criação do target RISCO_DEFASAGEM.
    # 
    # O target é criado usando:
    # - DEFASAGEM (diretamente nas regras)
    # - INDE (diretamente nas regras)
    # - INDICADORES_BAIXOS (diretamente nas regras)
    # - SCORE_ACADEMICO (diretamente nas regras)
    #
    # Para um modelo de produção válido, usamos apenas features que
    # são PREDITIVAS mas não DETERMINÍSTICAS do target.
    
    api_available_features = [
        # Dados demográficos e temporais (não usados nas regras)
        "FASE", "IDADE", "ANOS_PM", "VETERANO",
        # Indicadores individuais (INDE é limite, mas os componentes são úteis)
        "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV",
        # Flags comportamentais (não usados nas regras)
        "BOLSISTA", "PONTO_VIRADA",
        # Features estatísticas dos indicadores (NÃO incluir INDICADORES_BAIXOS)
        "MEDIA_INDICADORES", "STD_INDICADORES", "MIN_INDICADOR", "MAX_INDICADOR",
        "CV_INDICADORES", "RANGE_INDICADORES", "INDICADORES_ALTOS",
        # Features de interação (NÃO incluir SCORE_ACADEMICO)
        "RATIO_IDA_IEG", "RATIO_IPP_IPS", "ENGAJ_X_DESEMP",
        "SCORE_COMPORTAMENTAL", "INDE_X_ANOSPM", "BOLSISTA_E_VIRADA",
        # NOTA: Removidos para evitar data leakage:
        # - DEFASAGEM, DEFASAGEM_ABS, TEM_DEFASAGEM (usados na criação do target)
        # - INDICADORES_BAIXOS (usado na criação do target)
        # - SCORE_ACADEMICO, DIFF_ACAD_COMPORT (usados na criação do target)
        # - FASE_IDEAL (componente direto de DEFASAGEM)
        # - INDE (usado diretamente nas regras do target)
    ]
    
    # Filtrar apenas features disponíveis e numéricas
    feature_cols = [
        col for col in api_available_features
        if col in df.columns and df[col].dtype in ["int64", "float64", "int32", "float32"]
    ]
    
    logger.info(f"Features disponíveis para treinamento: {len(feature_cols)}")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 4.5. Tratar valores NaN restantes
    logger.info("4.5. Tratando valores NaN...")
    # Preencher NaN com mediana para numéricas
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Remover linhas onde y é NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Dados após tratamento de NaN: {len(X)} registros")
    
    # 5. Seleção de features
    logger.info("4. Selecionando features...")
    selector = FeatureSelector(method="importance", n_features=15)
    X = selector.fit_transform(X, y)
    
    # 5.5. Preencher NaN novamente após seleção (por segurança)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # 6. Split treino/teste
    logger.info("5. Dividindo dados...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAIN_CONFIG["test_size"],
        random_state=TRAIN_CONFIG["random_state"],
        stratify=y
    )
    
    # 7. Treinar modelo
    logger.info("6. Treinando modelo...")
    trainer = ModelTrainer(model_type="random_forest")
    trainer.train(X_train, y_train, optimize_hyperparams=True)
    
    # 8. Salvar modelo com nomes das features
    logger.info("7. Salvando modelo...")
    feature_names = X_train.columns.tolist()
    model_path = trainer.save(feature_names=feature_names)
    
    logger.info(f"Features do modelo: {feature_names}")
    logger.info(f"=== Pipeline concluído! Modelo salvo em {model_path} ===")
