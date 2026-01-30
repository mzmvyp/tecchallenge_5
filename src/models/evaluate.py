"""
Módulo de avaliação de modelos para o projeto Passos Mágicos.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import RISCO_LABELS, MONITORING_DIR
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)

# Tentar importar bibliotecas de visualização
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("matplotlib/seaborn não disponíveis para visualização")

# Tentar importar SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP não disponível para explicabilidade")


class ModelEvaluator:
    """Classe para avaliação de modelos de ML."""
    
    def __init__(self, model: Any, class_labels: Optional[Dict[int, str]] = None):
        """
        Inicializa o ModelEvaluator.
        
        Args:
            model: Modelo treinado (pipeline ou estimator)
            class_labels: Mapeamento de classes para labels
        """
        self.model = model
        self.class_labels = class_labels or RISCO_LABELS
        
        self.metrics: Dict[str, float] = {}
        self.predictions: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None
        
        logger.info("ModelEvaluator inicializado")
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Avalia o modelo nos dados fornecidos.
        
        Args:
            X: Features de teste
            y: Target de teste
            threshold: Threshold para classificação binária
        
        Returns:
            Dicionário com métricas
        """
        logger.info("Avaliando modelo...")
        
        # Converter para numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        self.y_true = y_array
        
        # Predições
        self.predictions = self.model.predict(X_array)
        
        # Probabilidades (se disponível)
        if hasattr(self.model, "predict_proba"):
            self.probabilities = self.model.predict_proba(X_array)
        
        # Calcular métricas
        self.metrics = self._calculate_metrics()
        
        logger.info(f"Avaliação concluída. Recall: {self.metrics['recall']:.4f}")
        
        return self.metrics
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calcula todas as métricas de avaliação."""
        metrics = {}
        
        # Métricas básicas
        metrics["accuracy"] = accuracy_score(self.y_true, self.predictions)
        
        # Para multiclasse, usar average='weighted' ou 'macro'
        n_classes = len(np.unique(self.y_true))
        average = "binary" if n_classes == 2 else "weighted"
        
        metrics["precision"] = precision_score(
            self.y_true, self.predictions, average=average, zero_division=0
        )
        metrics["recall"] = recall_score(
            self.y_true, self.predictions, average=average, zero_division=0
        )
        metrics["f1"] = f1_score(
            self.y_true, self.predictions, average=average, zero_division=0
        )
        
        # Métricas por classe
        for avg in ["macro", "weighted"]:
            metrics[f"precision_{avg}"] = precision_score(
                self.y_true, self.predictions, average=avg, zero_division=0
            )
            metrics[f"recall_{avg}"] = recall_score(
                self.y_true, self.predictions, average=avg, zero_division=0
            )
            metrics[f"f1_{avg}"] = f1_score(
                self.y_true, self.predictions, average=avg, zero_division=0
            )
        
        # AUC-ROC (se probabilidades disponíveis)
        if self.probabilities is not None:
            try:
                if n_classes == 2:
                    metrics["roc_auc"] = roc_auc_score(
                        self.y_true, self.probabilities[:, 1]
                    )
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        self.y_true, self.probabilities, 
                        multi_class="ovr", average="weighted"
                    )
            except Exception as e:
                logger.warning(f"Erro ao calcular AUC-ROC: {e}")
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Retorna matriz de confusão.
        
        Returns:
            Matriz de confusão
        """
        if self.predictions is None:
            raise ValueError("Execute evaluate() primeiro")
        
        return confusion_matrix(self.y_true, self.predictions)
    
    def get_classification_report(self) -> str:
        """
        Retorna relatório de classificação.
        
        Returns:
            String com relatório
        """
        if self.predictions is None:
            raise ValueError("Execute evaluate() primeiro")
        
        target_names = [self.class_labels[i] for i in sorted(self.class_labels.keys())]
        
        return classification_report(
            self.y_true,
            self.predictions,
            target_names=target_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> Optional[Any]:
        """
        Plota matriz de confusão.
        
        Args:
            save_path: Caminho para salvar figura
            figsize: Tamanho da figura
        
        Returns:
            Figure (se plotting disponível)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting não disponível")
            return None
        
        cm = self.get_confusion_matrix()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [self.class_labels[i] for i in sorted(self.class_labels.keys())]
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusão")
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Figura salva em: {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> Optional[Any]:
        """
        Plota curva ROC.
        
        Args:
            save_path: Caminho para salvar figura
            figsize: Tamanho da figura
        
        Returns:
            Figure (se plotting disponível)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting não disponível")
            return None
        
        if self.probabilities is None:
            logger.warning("Probabilidades não disponíveis para curva ROC")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        n_classes = self.probabilities.shape[1]
        
        # Plotar curva para cada classe
        for i in range(n_classes):
            y_binary = (self.y_true == i).astype(int)
            
            if y_binary.sum() == 0:
                continue
            
            fpr, tpr, _ = roc_curve(y_binary, self.probabilities[:, i])
            auc = roc_auc_score(y_binary, self.probabilities[:, i])
            
            label = f"{self.class_labels.get(i, f'Classe {i}')} (AUC = {auc:.3f})"
            ax.plot(fpr, tpr, label=label)
        
        # Linha diagonal (classificador aleatório)
        ax.plot([0, 1], [0, 1], "k--", label="Aleatório")
        
        ax.set_xlabel("Taxa de Falso Positivo")
        ax.set_ylabel("Taxa de Verdadeiro Positivo")
        ax.set_title("Curva ROC")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Figura salva em: {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 15,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[Any]:
        """
        Plota importância das features.
        
        Args:
            feature_names: Nomes das features
            top_n: Número de features a exibir
            save_path: Caminho para salvar
            figsize: Tamanho da figura
        
        Returns:
            Figure (se plotting disponível)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting não disponível")
            return None
        
        # Obter modelo interno do pipeline
        model = self.model
        if hasattr(model, "named_steps"):
            model = model.named_steps.get("model", model)
        
        # Verificar se tem importâncias
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()
        else:
            logger.warning("Modelo não suporta importância de features")
            return None
        
        # Criar DataFrame de importâncias
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            palette="viridis",
            ax=ax
        )
        
        ax.set_xlabel("Importância")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_n} Features mais Importantes")
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Figura salva em: {save_path}")
        
        return fig
    
    def explain_with_shap(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        max_samples: int = 100
    ) -> Optional[Any]:
        """
        Gera explicações SHAP para as predições.
        
        Args:
            X: Features para explicar
            feature_names: Nomes das features
            max_samples: Máximo de amostras para calcular SHAP
        
        Returns:
            SHAP explainer (se disponível)
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP não disponível")
            return None
        
        logger.info("Calculando valores SHAP...")
        
        # Obter modelo interno
        model = self.model
        if hasattr(model, "named_steps"):
            model = model.named_steps.get("model", model)
        
        # Limitar amostras
        if len(X) > max_samples:
            X_sample = X[:max_samples] if isinstance(X, np.ndarray) else X.iloc[:max_samples]
        else:
            X_sample = X
        
        # Criar explainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            logger.info("Valores SHAP calculados com sucesso")
            
            return {
                "explainer": explainer,
                "shap_values": shap_values,
                "feature_names": feature_names
            }
        except Exception as e:
            logger.error(f"Erro ao calcular SHAP: {e}")
            return None
    
    def generate_report(
        self,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Gera relatório completo de avaliação.
        
        Args:
            output_dir: Diretório para salvar relatório
        
        Returns:
            Dicionário com todos os resultados
        """
        if output_dir is None:
            output_dir = MONITORING_DIR / "evaluation"
        
        ensure_dir(output_dir)
        
        report = {
            "metrics": self.metrics,
            "classification_report": self.get_classification_report(),
            "confusion_matrix": self.get_confusion_matrix().tolist(),
        }
        
        # Salvar figuras
        if PLOTTING_AVAILABLE:
            self.plot_confusion_matrix(save_path=output_dir / "confusion_matrix.png")
            self.plot_roc_curve(save_path=output_dir / "roc_curve.png")
        
        # Salvar métricas em JSON
        import json
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Relatório salvo em: {output_dir}")
        
        return report


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Função de conveniência para avaliar modelo.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
    
    Returns:
        Dicionário com métricas
    """
    evaluator = ModelEvaluator(model)
    return evaluator.evaluate(X_test, y_test)


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compara múltiplos modelos.
    
    Args:
        models: Dicionário {nome: modelo}
        X_test: Features de teste
        y_test: Target de teste
    
    Returns:
        DataFrame com métricas de todos os modelos
    """
    results = []
    
    for name, model in models.items():
        logger.info(f"Avaliando {name}...")
        
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)
        metrics["model"] = name
        results.append(metrics)
    
    return pd.DataFrame(results).set_index("model")
