# Referência da API

## Base URL

```
http://localhost:8000
```

## Autenticação

A API atualmente não requer autenticação. Em produção, recomenda-se implementar autenticação via API Key ou OAuth2.

## Endpoints

### Health Check

#### GET /health

Verifica o status de saúde da API.

**Resposta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Predições

#### POST /predict

Realiza predição de risco de defasagem para um estudante.

**Request Body:**
```json
{
  "fase": 5,
  "idade": 12,
  "genero": "M",
  "anos_na_pm": 2,
  "inde": 6.5,
  "ian": 7.0,
  "ida": 6.0,
  "ieg": 7.5,
  "iaa": 7.0,
  "ips": 6.5,
  "ipp": 7.0,
  "ipv": 6.0,
  "pedra": "Ágata",
  "instituicao_ensino": "Escola Municipal",
  "bolsista": false,
  "ponto_virada": false
}
```

**Parâmetros:**

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| fase | int | Sim | Fase atual (1-8) |
| idade | int | Sim | Idade do estudante (6-25) |
| genero | string | Sim | Gênero ("M" ou "F") |
| anos_na_pm | int | Sim | Anos na Passos Mágicos |
| inde | float | Sim | INDE (0-10) |
| ian | float | Sim | IAN (0-10) |
| ida | float | Sim | IDA (0-10) |
| ieg | float | Sim | IEG (0-10) |
| iaa | float | Sim | IAA (0-10) |
| ips | float | Sim | IPS (0-10) |
| ipp | float | Sim | IPP (0-10) |
| ipv | float | Sim | IPV (0-10) |
| pedra | string | Sim | Classificação PEDRA |
| instituicao_ensino | string | Sim | Instituição de ensino |
| bolsista | boolean | Sim | Se é bolsista |
| ponto_virada | boolean | Não | Se atingiu ponto de virada |

**Resposta:**
```json
{
  "risco_defasagem": 1,
  "probabilidade": 0.65,
  "nivel_risco": "MÉDIO",
  "probabilidades_por_classe": {
    "BAIXO": 0.25,
    "MÉDIO": 0.65,
    "ALTO": 0.10
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

#### POST /predict/batch

Realiza predição em lote para múltiplos estudantes.

**Request Body:**
```json
{
  "students": [
    { ... },
    { ... }
  ]
}
```

**Resposta:**
```json
{
  "predictions": [...],
  "total": 2,
  "processing_time_ms": 150.5
}
```

### Monitoramento

#### GET /monitoring/metrics

Retorna métricas agregadas do sistema.

**Resposta:**
```json
{
  "total_predictions": 1500,
  "total_errors": 5,
  "error_rate": 0.003,
  "prediction_distribution": {
    "BAIXO": 800,
    "MÉDIO": 500,
    "ALTO": 200
  },
  "latency": {
    "mean_ms": 45.2,
    "p95_ms": 120.5
  },
  "predictions_per_minute": 25.0,
  "session_uptime_seconds": 3600
}
```

## Códigos de Erro

| Código | Descrição |
|--------|-----------|
| 400 | Bad Request - Dados inválidos |
| 422 | Validation Error - Falha na validação |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Modelo não carregado |
