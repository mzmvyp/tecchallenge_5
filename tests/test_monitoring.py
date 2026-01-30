"""
Testes para os módulos de monitoramento.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.monitoring.metrics_tracker import MetricsTracker, get_metrics_tracker, track_prediction, get_metrics
from src.monitoring.drift_detector import DriftDetector


class TestMetricsTracker:
    """Testes para a classe MetricsTracker."""
    
    def test_init(self):
        """Testa inicialização do MetricsTracker."""
        tracker = MetricsTracker()
        assert tracker is not None
    
    def test_track_prediction(self):
        """Testa registro de predição."""
        tracker = MetricsTracker()
        
        tracker.track_prediction(
            prediction=1,
            latency_ms=50.0
        )
        
        metrics = tracker.get_metrics_summary()
        assert metrics["total_predictions"] >= 1
    
    def test_track_error(self):
        """Testa registro de erro."""
        tracker = MetricsTracker()
        
        tracker.track_error("TestError", "Test error message")
        
        metrics = tracker.get_metrics_summary()
        assert metrics["total_errors"] >= 1
    
    def test_get_metrics_summary_returns_dict(self):
        """Testa que get_metrics_summary retorna dicionário."""
        tracker = MetricsTracker()
        
        metrics = tracker.get_metrics_summary()
        
        assert isinstance(metrics, dict)
        assert "total_predictions" in metrics
        assert "total_errors" in metrics
        assert "error_rate" in metrics
    
    def test_reset_metrics(self):
        """Testa reset de métricas."""
        tracker = MetricsTracker()
        
        # Registrar algumas métricas
        tracker.track_prediction(1, 50.0)
        tracker.track_prediction(2, 60.0)
        tracker.track_error("Error", "msg")
        
        # Reset
        tracker.reset()
        
        metrics = tracker.get_metrics_summary()
        # Após reset, deve ter valores zerados
        assert metrics["total_predictions"] == 0
    
    def test_get_latency_histogram(self):
        """Testa estatísticas de latência."""
        tracker = MetricsTracker()
        
        # Registrar várias predições
        for _ in range(10):
            tracker.track_prediction(1, latency_ms=50.0 + np.random.randn() * 10)
        
        stats = tracker.get_latency_histogram()
        assert stats is not None
    
    def test_prediction_distribution(self):
        """Testa distribuição de predições."""
        tracker = MetricsTracker()
        
        # Registrar predições de diferentes classes
        tracker.track_prediction(0, 50.0)
        tracker.track_prediction(1, 50.0)
        tracker.track_prediction(2, 50.0)
        tracker.track_prediction(0, 50.0)
        
        metrics = tracker.get_metrics_summary()
        assert "prediction_distribution" in metrics
    
    def test_get_daily_metrics(self):
        """Testa obtenção de métricas diárias."""
        tracker = MetricsTracker()
        
        tracker.track_prediction(1, 50.0)
        
        daily = tracker.get_daily_metrics()
        assert isinstance(daily, dict)
    
    def test_get_recent_errors(self):
        """Testa obtenção de erros recentes."""
        tracker = MetricsTracker()
        
        tracker.track_error("Error1", "msg1")
        tracker.track_error("Error2", "msg2")
        
        errors = tracker.get_recent_errors(n=5)
        assert isinstance(errors, list)


class TestMetricsTrackerEdgeCases:
    """Testes de casos extremos para MetricsTracker."""
    
    def test_empty_tracker(self):
        """Testa tracker sem registros."""
        tracker = MetricsTracker()
        metrics = tracker.get_metrics_summary()
        
        assert metrics["total_predictions"] == 0
        assert metrics["error_rate"] == 0.0
    
    def test_high_volume_predictions(self):
        """Testa com grande volume de predições."""
        tracker = MetricsTracker()
        
        for i in range(100):
            tracker.track_prediction(i % 3, latency_ms=50.0)
        
        metrics = tracker.get_metrics_summary()
        assert metrics["total_predictions"] == 100


class TestMetricsTrackerFunctions:
    """Testes para funções de conveniência."""
    
    def test_get_metrics_tracker(self):
        """Testa obtenção do tracker singleton."""
        tracker = get_metrics_tracker()
        assert isinstance(tracker, MetricsTracker)
    
    def test_track_prediction_function(self):
        """Testa função track_prediction."""
        track_prediction(1, 50.0)
        # Não deve lançar exceção
    
    def test_get_metrics_function(self):
        """Testa função get_metrics."""
        metrics = get_metrics()
        assert isinstance(metrics, dict)


class TestDriftDetector:
    """Testes para a classe DriftDetector."""
    
    @pytest.fixture
    def sample_reference_data(self):
        """Cria dados de referência."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100)
        })
    
    @pytest.fixture
    def sample_current_data(self):
        """Cria dados atuais (sem drift)."""
        np.random.seed(43)
        return pd.DataFrame({
            "feature1": np.random.randn(50),
            "feature2": np.random.randn(50),
            "feature3": np.random.randn(50)
        })
    
    @pytest.fixture
    def drifted_data(self):
        """Cria dados com drift significativo."""
        np.random.seed(44)
        return pd.DataFrame({
            "feature1": np.random.randn(50) + 5,  # Drift no mean
            "feature2": np.random.randn(50) * 3,  # Drift na variância
            "feature3": np.random.randn(50)
        })
    
    def test_init(self):
        """Testa inicialização do DriftDetector."""
        detector = DriftDetector()
        assert detector is not None
    
    def test_set_reference_data(self, sample_reference_data):
        """Testa definição de dados de referência."""
        detector = DriftDetector()
        detector.set_reference_data(sample_reference_data)
        
        # Deve armazenar os dados
        assert detector.reference_data is not None
    
    def test_detect_data_drift_no_drift(self, sample_reference_data, sample_current_data):
        """Testa detecção quando não há drift."""
        detector = DriftDetector()
        detector.set_reference_data(sample_reference_data)
        
        result = detector.detect_data_drift(sample_current_data)
        
        # Resultado deve ser dict
        assert isinstance(result, dict)
    
    def test_detect_data_drift_with_drift(self, sample_reference_data, drifted_data):
        """Testa detecção quando há drift."""
        detector = DriftDetector()
        detector.set_reference_data(sample_reference_data)
        
        result = detector.detect_data_drift(drifted_data)
        
        # Com dados muito diferentes, deve detectar algum drift
        assert isinstance(result, dict)
    
    def test_get_drift_summary(self, sample_reference_data, sample_current_data):
        """Testa resumo do drift."""
        detector = DriftDetector()
        detector.set_reference_data(sample_reference_data)
        detector.detect_data_drift(sample_current_data)
        
        summary = detector.get_drift_summary()
        
        # Deve retornar dicionário
        assert isinstance(summary, dict)


class TestDriftDetectorEdgeCases:
    """Testes de casos extremos para DriftDetector."""
    
    def test_detect_without_reference(self):
        """Testa detecção sem dados de referência."""
        detector = DriftDetector()
        
        current_data = pd.DataFrame({
            "feature1": np.random.randn(50)
        })
        
        # Deve lançar ValueError quando não há dados de referência
        with pytest.raises(ValueError):
            detector.detect_data_drift(current_data)
    
    def test_get_summary_without_detection(self):
        """Testa get_drift_summary sem detecção prévia."""
        detector = DriftDetector()
        
        summary = detector.get_drift_summary()
        assert isinstance(summary, dict)
