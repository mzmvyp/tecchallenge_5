"""
Testes para o módulo de modelos.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.train import ModelTrainer, train_model
from src.models.evaluate import ModelEvaluator, evaluate_model
from src.models.predict import ModelPredictor


class TestModelTrainer:
    """Testes para a classe ModelTrainer."""
    
    def test_init_default_params(self):
        """Testa inicialização com parâmetros padrão."""
        trainer = ModelTrainer()
        
        assert trainer.model_type == "random_forest"
        assert trainer.cv_folds == 5
        assert trainer.scoring == "recall_weighted"  # Multiclasse usa recall_weighted
    
    def test_init_custom_params(self):
        """Testa inicialização com parâmetros customizados."""
        trainer = ModelTrainer(
            model_type="logistic_regression",
            cv_folds=3,
            scoring="f1"
        )
        
        assert trainer.model_type == "logistic_regression"
        assert trainer.cv_folds == 3
        assert trainer.scoring == "f1"
    
    def test_train_without_optimization(self, sample_features_df, sample_target):
        """Testa treinamento sem otimização."""
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(sample_features_df, sample_target, optimize_hyperparams=False)
        
        assert trainer.model is not None
        assert trainer.pipeline is not None
        assert "mean_score" in trainer.cv_results
    
    def test_train_with_optimization(self, sample_features_df, sample_target):
        """Testa treinamento com otimização de hiperparâmetros."""
        trainer = ModelTrainer(model_type="logistic_regression")
        trainer.train(sample_features_df, sample_target, optimize_hyperparams=True)
        
        assert trainer.model is not None
        assert len(trainer.best_params) > 0
    
    def test_train_different_models(self, sample_features_df, sample_target):
        """Testa treinamento com diferentes tipos de modelo."""
        model_types = ["logistic_regression", "random_forest", "gradient_boosting"]
        
        for model_type in model_types:
            trainer = ModelTrainer(model_type=model_type)
            trainer.train(sample_features_df, sample_target, optimize_hyperparams=False)
            
            assert trainer.model is not None
    
    def test_save_model(self, sample_features_df, sample_target, temp_model_path):
        """Testa salvamento do modelo."""
        trainer = ModelTrainer()
        trainer.train(sample_features_df, sample_target, optimize_hyperparams=False)
        
        saved_path = trainer.save(temp_model_path)
        
        assert saved_path.exists()
    
    def test_save_without_training_raises_error(self, temp_model_path):
        """Testa que save sem treinar levanta erro."""
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError, match="Nenhum modelo treinado"):
            trainer.save(temp_model_path)
    
    def test_get_feature_importances(self, sample_features_df, sample_target):
        """Testa obtenção de importâncias de features."""
        trainer = ModelTrainer(model_type="random_forest")
        trainer.train(sample_features_df, sample_target, optimize_hyperparams=False)
        
        importances = trainer.get_feature_importances(list(sample_features_df.columns))
        
        assert len(importances) == len(sample_features_df.columns)
        assert all(v >= 0 for v in importances.values)


class TestModelEvaluator:
    """Testes para a classe ModelEvaluator."""
    
    def test_evaluate_returns_metrics(self, mock_model, sample_features_df, sample_target):
        """Testa que evaluate retorna métricas."""
        evaluator = ModelEvaluator(mock_model)
        metrics = evaluator.evaluate(sample_features_df, sample_target)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
    
    def test_metrics_in_valid_range(self, mock_model, sample_features_df, sample_target):
        """Testa que métricas estão em range válido."""
        evaluator = ModelEvaluator(mock_model)
        metrics = evaluator.evaluate(sample_features_df, sample_target)
        
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} fora do range: {value}"
    
    def test_get_confusion_matrix(self, mock_model, sample_features_df, sample_target):
        """Testa obtenção de matriz de confusão."""
        evaluator = ModelEvaluator(mock_model)
        evaluator.evaluate(sample_features_df, sample_target)
        
        cm = evaluator.get_confusion_matrix()
        
        assert cm is not None
        assert cm.shape[0] == cm.shape[1]  # Matriz quadrada
    
    def test_get_classification_report(self, mock_model, sample_features_df, sample_target):
        """Testa obtenção de relatório de classificação."""
        evaluator = ModelEvaluator(mock_model)
        evaluator.evaluate(sample_features_df, sample_target)
        
        report = evaluator.get_classification_report()
        
        assert isinstance(report, str)
        assert "precision" in report.lower()
        assert "recall" in report.lower()
    
    def test_evaluate_model_convenience(self, mock_model, sample_features_df, sample_target):
        """Testa função de conveniência evaluate_model."""
        metrics = evaluate_model(mock_model, sample_features_df, sample_target)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics


class TestModelPredictor:
    """Testes para a classe ModelPredictor."""
    
    @pytest.fixture
    def trained_model_path(self, sample_features_df, sample_target, temp_model_path):
        """Fixture que cria modelo treinado e retorna caminho."""
        trainer = ModelTrainer()
        trainer.train(sample_features_df, sample_target, optimize_hyperparams=False)
        trainer.save(temp_model_path)
        return temp_model_path
    
    def test_load_model(self, trained_model_path):
        """Testa carregamento do modelo."""
        predictor = ModelPredictor(trained_model_path)
        
        assert predictor.model is not None
        assert predictor.pipeline is not None
    
    def test_load_nonexistent_model_raises_error(self, tmp_path):
        """Testa que carregar modelo inexistente levanta erro."""
        fake_path = tmp_path / "nonexistent.joblib"
        
        with pytest.raises(FileNotFoundError):
            ModelPredictor(fake_path)
    
    def test_predict(self, trained_model_path, sample_features_df):
        """Testa predição."""
        predictor = ModelPredictor(trained_model_path)
        predictions = predictor.predict(sample_features_df.head())
        
        assert len(predictions) == 5
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_predict_proba(self, trained_model_path, sample_features_df):
        """Testa predição de probabilidades."""
        predictor = ModelPredictor(trained_model_path)
        probas = predictor.predict_proba(sample_features_df.head())
        
        assert probas.shape[0] == 5
        assert probas.shape[1] == 3  # 3 classes
        # Probabilidades devem somar 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(5))
    
    def test_predict_single(self, trained_model_path):
        """Testa predição individual."""
        predictor = ModelPredictor(trained_model_path)
        
        data = {
            "FASE": 5,
            "IDADE": 12,
            "INDE": 6.5,
            "IAN": 7.0,
            "IDA": 6.0,
            "IEG": 7.5,
            "IAA": 7.0,
            "IPS": 6.5,
            "IPP": 7.0,
            "IPV": 6.0
        }
        
        result = predictor.predict_single(data)
        
        assert "risco_defasagem" in result
        assert "nivel_risco" in result
        assert "probabilidade" in result
        assert result["risco_defasagem"] in [0, 1, 2]
    
    def test_predict_batch(self, trained_model_path):
        """Testa predição em lote."""
        predictor = ModelPredictor(trained_model_path)
        
        data = [
            {"FASE": 5, "IDADE": 12, "INDE": 6.5, "IAN": 7.0, "IDA": 6.0,
             "IEG": 7.5, "IAA": 7.0, "IPS": 6.5, "IPP": 7.0, "IPV": 6.0},
            {"FASE": 3, "IDADE": 10, "INDE": 5.0, "IAN": 5.5, "IDA": 4.5,
             "IEG": 5.0, "IAA": 5.0, "IPS": 5.0, "IPP": 5.0, "IPV": 5.0},
        ]
        
        results = predictor.predict_batch(data)
        
        assert len(results) == 2
        assert all("risco_defasagem" in r for r in results)
    
    def test_get_model_info(self, trained_model_path):
        """Testa obtenção de informações do modelo."""
        predictor = ModelPredictor(trained_model_path)
        info = predictor.get_model_info()
        
        assert "model_path" in info
        assert "model_type" in info
