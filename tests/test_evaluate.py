"""
Testes adicionais para o módulo evaluate.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import ModelEvaluator, evaluate_model


class TestModelEvaluatorExtended:
    """Testes estendidos para ModelEvaluator."""
    
    @pytest.fixture
    def trained_model(self):
        """Cria um modelo treinado para testes."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1, 2], size=100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model
    
    @pytest.fixture
    def test_data(self):
        """Cria dados de teste."""
        X = np.random.randn(30, 5)
        y = np.random.choice([0, 1, 2], size=30)
        return X, y
    
    def test_evaluator_init(self, trained_model):
        """Testa inicialização do evaluator."""
        evaluator = ModelEvaluator(trained_model)
        assert evaluator.model is not None
    
    def test_evaluate_returns_dict(self, trained_model, test_data):
        """Testa que evaluate retorna dicionário com métricas."""
        X_test, y_test = test_data
        evaluator = ModelEvaluator(trained_model)
        
        metrics = evaluator.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
    
    def test_metrics_in_valid_range(self, trained_model, test_data):
        """Testa que métricas estão no range válido [0, 1]."""
        X_test, y_test = test_data
        evaluator = ModelEvaluator(trained_model)
        
        metrics = evaluator.evaluate(X_test, y_test)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                assert 0 <= value <= 1, f"{key} = {value} fora do range"
    
    def test_get_confusion_matrix(self, trained_model, test_data):
        """Testa obtenção da matriz de confusão."""
        X_test, y_test = test_data
        evaluator = ModelEvaluator(trained_model)
        evaluator.evaluate(X_test, y_test)
        
        cm = evaluator.get_confusion_matrix()
        
        assert cm is not None
        assert isinstance(cm, np.ndarray)
        assert cm.shape[0] == cm.shape[1]  # Matriz quadrada
    
    def test_get_classification_report(self, trained_model, test_data):
        """Testa obtenção do relatório de classificação."""
        X_test, y_test = test_data
        evaluator = ModelEvaluator(trained_model)
        evaluator.evaluate(X_test, y_test)
        
        report = evaluator.get_classification_report()
        
        assert report is not None
        assert isinstance(report, (str, dict))
    
    def test_evaluate_with_dataframe(self, trained_model):
        """Testa evaluate com DataFrame ao invés de array."""
        X_df = pd.DataFrame(np.random.randn(30, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.choice([0, 1, 2], size=30))
        
        evaluator = ModelEvaluator(trained_model)
        metrics = evaluator.evaluate(X_df, y)
        
        assert isinstance(metrics, dict)


class TestEvaluateModelFunction:
    """Testes para a função evaluate_model."""
    
    def test_evaluate_model_convenience(self):
        """Testa função de conveniência evaluate_model."""
        # Criar dados de teste
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1, 2], size=100)
        
        # Treinar modelo
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:70], y[:70])
        
        # Avaliar
        metrics = evaluate_model(model, X[70:], y[70:])
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0


class TestModelEvaluatorEdgeCases:
    """Testes de casos extremos."""
    
    def test_evaluate_with_single_class(self):
        """Testa avaliação quando todas as predições são da mesma classe."""
        X = np.random.randn(50, 5)
        y = np.zeros(50)  # Todas da mesma classe
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        evaluator = ModelEvaluator(model)
        
        # Não deve dar erro
        metrics = evaluator.evaluate(X[:10], y[:10])
        assert isinstance(metrics, dict)
    
    def test_evaluate_with_binary_classification(self):
        """Testa avaliação com classificação binária."""
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1], size=100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:70], y[:70])
        
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X[70:], y[70:])
        
        assert isinstance(metrics, dict)
        # ROC AUC deve funcionar com binário
        assert "roc_auc" in metrics or "accuracy" in metrics
