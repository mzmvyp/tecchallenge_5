"""
Testes para a API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import após configurar path
from api.main import app
from api.schemas import StudentInput, PredictionOutput


@pytest.fixture
def client():
    """Cliente de teste para a API."""
    return TestClient(app)


@pytest.fixture
def mock_predictor():
    """Mock do predictor para testes."""
    mock = Mock()
    mock.predict_risk.return_value = {
        "risco_defasagem": 1,
        "nivel_risco": "MÉDIO",
        "probabilidade": 0.65,
        "probabilidades_por_classe": {
            "BAIXO": 0.25,
            "MÉDIO": 0.65,
            "ALTO": 0.10
        },
        "timestamp": "2024-01-15T10:30:00"
    }
    return mock


class TestHealthEndpoints:
    """Testes para endpoints de health."""
    
    def test_health_check(self, client):
        """Testa endpoint /health."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
    
    def test_liveness_check(self, client):
        """Testa endpoint /health/live."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
    
    def test_readiness_check(self, client):
        """Testa endpoint /health/ready."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data


class TestPredictEndpoints:
    """Testes para endpoints de predição."""
    
    def test_predict_valid_input(self, client, api_student_input, mock_predictor):
        """Testa predição com entrada válida."""
        with patch("api.routes.predict.get_predictor", return_value=mock_predictor):
            response = client.post("/predict", json=api_student_input)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "risco_defasagem" in data
        assert "nivel_risco" in data
        assert "probabilidade" in data
        assert data["risco_defasagem"] in [0, 1, 2]
    
    def test_predict_invalid_fase(self, client, api_student_input):
        """Testa predição com fase inválida."""
        api_student_input["fase"] = 10  # Inválido (máximo é 8)
        
        response = client.post("/predict", json=api_student_input)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_inde(self, client, api_student_input):
        """Testa predição com INDE inválido."""
        api_student_input["inde"] = 15  # Inválido (máximo é 10)
        
        response = client.post("/predict", json=api_student_input)
        
        assert response.status_code == 422
    
    def test_predict_missing_field(self, client, api_student_input):
        """Testa predição com campo faltando."""
        del api_student_input["inde"]
        
        response = client.post("/predict", json=api_student_input)
        
        assert response.status_code == 422
    
    def test_predict_batch(self, client, api_student_input, mock_predictor):
        """Testa predição em lote."""
        batch_input = {"students": [api_student_input, api_student_input]}
        
        with patch("api.routes.predict.get_predictor", return_value=mock_predictor):
            response = client.post("/predict/batch", json=batch_input)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert "total" in data
        assert data["total"] == 2
    
    def test_predict_batch_empty(self, client):
        """Testa predição em lote vazio."""
        response = client.post("/predict/batch", json={"students": []})
        
        assert response.status_code == 422  # Validation error (min_length=1)
    
    def test_risk_levels_endpoint(self, client):
        """Testa endpoint de níveis de risco."""
        response = client.get("/predict/risk-levels")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "levels" in data
        assert len(data["levels"]) == 3


class TestMonitoringEndpoints:
    """Testes para endpoints de monitoramento."""
    
    def test_get_metrics(self, client):
        """Testa endpoint /monitoring/metrics."""
        response = client.get("/monitoring/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_predictions" in data
        assert "total_errors" in data
        assert "error_rate" in data
    
    def test_get_daily_metrics(self, client):
        """Testa endpoint /monitoring/metrics/daily."""
        response = client.get("/monitoring/metrics/daily")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "date" in data
        assert "total_predictions" in data
    
    def test_get_latency_histogram(self, client):
        """Testa endpoint /monitoring/metrics/latency."""
        response = client.get("/monitoring/metrics/latency?bins=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "counts" in data
        assert "edges" in data
    
    def test_get_recent_errors(self, client):
        """Testa endpoint /monitoring/errors."""
        response = client.get("/monitoring/errors?n=5")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total" in data
        assert "errors" in data
    
    def test_reset_metrics(self, client):
        """Testa endpoint de reset de métricas."""
        response = client.post("/monitoring/metrics/reset")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"


class TestRootEndpoint:
    """Testes para endpoint raiz."""
    
    def test_root(self, client):
        """Testa endpoint raiz."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestSchemas:
    """Testes para schemas Pydantic."""
    
    def test_student_input_valid(self, api_student_input):
        """Testa criação de StudentInput válido."""
        student = StudentInput(**api_student_input)
        
        assert student.fase == 5
        assert student.inde == 6.5
    
    def test_student_input_genero_uppercase(self, api_student_input):
        """Testa que gênero é convertido para uppercase."""
        api_student_input["genero"] = "m"
        student = StudentInput(**api_student_input)
        
        assert student.genero == "M"
    
    def test_student_input_pedra_titlecase(self, api_student_input):
        """Testa que pedra é convertido para titlecase."""
        api_student_input["pedra"] = "ÁGATA"
        student = StudentInput(**api_student_input)
        
        assert student.pedra == "Ágata"
    
    def test_student_input_invalid_genero(self, api_student_input):
        """Testa validação de gênero inválido."""
        api_student_input["genero"] = "X"
        
        with pytest.raises(ValueError):
            StudentInput(**api_student_input)
    
    def test_student_input_invalid_pedra(self, api_student_input):
        """Testa validação de pedra inválida."""
        api_student_input["pedra"] = "Diamante"
        
        with pytest.raises(ValueError):
            StudentInput(**api_student_input)
