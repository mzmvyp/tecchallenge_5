# Guia de Deployment

## Opções de Deploy

### 1. Desenvolvimento Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar modelo (se necessário)
python -m src.models.train

# Iniciar API
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Docker (Recomendado)

#### Build da Imagem

```bash
docker build -t passos-magicos-ml:latest .
```

#### Executar Container

```bash
docker run -d \
  --name passos-magicos-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e MODEL_PATH=/app/models/model.joblib \
  -e LOG_LEVEL=INFO \
  passos-magicos-ml:latest
```

#### Docker Compose

```bash
docker-compose up -d
```

### 3. Kubernetes

#### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: passos-magicos-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: passos-magicos-api
  template:
    metadata:
      labels:
        app: passos-magicos-api
    spec:
      containers:
      - name: api
        image: passos-magicos-ml:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: /app/models/model.joblib
        - name: LOG_LEVEL
          value: INFO
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

#### Service YAML

```yaml
apiVersion: v1
kind: Service
metadata:
  name: passos-magicos-service
spec:
  selector:
    app: passos-magicos-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| ENVIRONMENT | Ambiente (development/production) | development |
| DEBUG | Modo debug | True |
| MODEL_PATH | Caminho do modelo | models/model.joblib |
| MODEL_VERSION | Versão do modelo | 1.0.0 |
| API_HOST | Host da API | 0.0.0.0 |
| API_PORT | Porta da API | 8000 |
| LOG_LEVEL | Nível de log | INFO |
| LOG_FILE | Arquivo de log | logs/app.log |

## Health Checks

### Endpoints de Saúde

- `/health` - Status geral
- `/health/ready` - Readiness (modelo carregado)
- `/health/live` - Liveness (aplicação rodando)

### Configuração de Probes

```python
# Kubernetes
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5

livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 10
```

## Monitoramento em Produção

### Logs

Logs são salvos em formato estruturado:

```
2024-01-15 10:30:00 | INFO | api.main | Requisição: POST /predict
2024-01-15 10:30:00 | INFO | api.main | Resposta: 200 | Duração: 45.2ms
```

### Métricas

Acessíveis via `/monitoring/metrics`:

- Total de predições
- Taxa de erros
- Latência (média, p95, p99)
- Distribuição de classes

### Drift Detection

Execute periodicamente para detectar drift nos dados:

```python
from src.monitoring import DriftDetector

detector = DriftDetector(reference_data=df_train)
results = detector.detect_data_drift(df_production)

if results["drift_detected"]:
    # Alertar equipe e considerar retreinamento
    pass
```

## Segurança

### Recomendações para Produção

1. **Autenticação**: Implementar API Key ou OAuth2
2. **Rate Limiting**: Limitar requisições por IP
3. **HTTPS**: Usar certificado SSL/TLS
4. **CORS**: Restringir origens permitidas
5. **Secrets**: Usar secret manager para credenciais

### Exemplo com API Key

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
```

## Backup e Recuperação

### Modelo

- Versionamento com MLflow ou DVC
- Armazenamento em S3/GCS/Azure Blob
- Rollback automatizado se métricas degradarem

### Dados

- Backup diário dos dados de treino
- Logs de predições para auditoria
- Retenção conforme política de dados
