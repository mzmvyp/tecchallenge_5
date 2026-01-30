"""
Testes estendidos para a API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app


client = TestClient(app)


class TestAPIRoot:
    """Testes para o endpoint raiz."""
    
    def test_root_returns_info(self):
        """Testa que root retorna informações do app."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestPredictEdgeCases:
    """Testes de casos extremos para /predict."""
    
    def test_predict_with_minimum_values(self):
        """Testa predição com valores mínimos."""
        payload = {
            "fase": 1,
            "idade": 6,
            "genero": "F",
            "anos_na_pm": 0,
            "inde": 0.0,
            "ian": 0.0,
            "ida": 0.0,
            "ieg": 0.0,
            "iaa": 0.0,
            "ips": 0.0,
            "ipp": 0.0,
            "ipv": 0.0,
            "pedra": "Quartzo",
            "instituicao_ensino": "Escola",
            "bolsista": False,
            "ponto_virada": False
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
    
    def test_predict_with_maximum_values(self):
        """Testa predição com valores máximos."""
        payload = {
            "fase": 8,
            "idade": 25,
            "genero": "M",
            "anos_na_pm": 15,
            "inde": 10.0,
            "ian": 10.0,
            "ida": 10.0,
            "ieg": 10.0,
            "iaa": 10.0,
            "ips": 10.0,
            "ipp": 10.0,
            "ipv": 10.0,
            "pedra": "Topázio",
            "instituicao_ensino": "Escola Top",
            "bolsista": True,
            "ponto_virada": True
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


class TestHealthExtended:
    """Testes estendidos para health check."""
    
    def test_health_contains_required_fields(self):
        """Testa que health contém campos obrigatórios."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
    
    def test_liveness_simple(self):
        """Testa liveness probe simples."""
        response = client.get("/health/live")
        assert response.status_code == 200
    
    def test_readiness_simple(self):
        """Testa readiness probe simples."""
        response = client.get("/health/ready")
        assert response.status_code == 200


class TestMonitoringExtended:
    """Testes estendidos para monitoramento."""
    
    def test_metrics_endpoint(self):
        """Testa endpoint de métricas."""
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
    
    def test_daily_metrics_endpoint(self):
        """Testa endpoint de métricas diárias."""
        response = client.get("/monitoring/metrics/daily")
        assert response.status_code == 200
    
    def test_latency_histogram_endpoint(self):
        """Testa endpoint de histograma de latência."""
        response = client.get("/monitoring/metrics/latency")
        assert response.status_code == 200
    
    def test_errors_endpoint(self):
        """Testa endpoint de erros."""
        response = client.get("/monitoring/errors")
        assert response.status_code == 200


class TestBatchPrediction:
    """Testes para predição em lote."""
    
    def test_batch_single_student(self):
        """Testa batch com único estudante."""
        payload = {
            "students": [{
                "fase": 5,
                "idade": 12,
                "genero": "M",
                "anos_na_pm": 2,
                "inde": 7.0,
                "ian": 7.0,
                "ida": 7.0,
                "ieg": 7.0,
                "iaa": 7.0,
                "ips": 7.0,
                "ipp": 7.0,
                "ipv": 7.0,
                "pedra": "Ametista",
                "instituicao_ensino": "Escola",
                "bolsista": False,
                "ponto_virada": False
            }]
        }
        
        response = client.post("/predict/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
    
    def test_batch_multiple_students(self):
        """Testa batch com múltiplos estudantes."""
        base_student = {
            "fase": 5,
            "idade": 12,
            "genero": "M",
            "anos_na_pm": 2,
            "inde": 7.0,
            "ian": 7.0,
            "ida": 7.0,
            "ieg": 7.0,
            "iaa": 7.0,
            "ips": 7.0,
            "ipp": 7.0,
            "ipv": 7.0,
            "pedra": "Ametista",
            "instituicao_ensino": "Escola",
            "bolsista": False,
            "ponto_virada": False
        }
        
        payload = {
            "students": [base_student, base_student, base_student]
        }
        
        response = client.post("/predict/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3


class TestRiskLevels:
    """Testes para endpoint de níveis de risco."""
    
    def test_risk_levels_returns_all_levels(self):
        """Testa que risk_levels retorna todos os níveis."""
        response = client.get("/predict/risk-levels")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "0" in data or "BAIXO" in str(data)
        assert "1" in data or "MEDIO" in str(data) or "MÉDIO" in str(data)
        assert "2" in data or "ALTO" in str(data)
