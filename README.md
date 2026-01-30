# 🎓 Passos Mágicos ML - Predição de Risco de Defasagem Escolar

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Projeto de Machine Learning desenvolvido para a **Associação Passos Mágicos**, uma ONG que transforma a vida de crianças e jovens de baixa renda através da educação em Embu-Guaçu/SP.

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [API](#-api)
- [Pipeline de ML](#-pipeline-de-ml)
- [Monitoramento](#-monitoramento)
- [Testes](#-testes)
- [Docker](#-docker)
- [Contribuição](#-contribuição)

## 🎯 Visão Geral

### Problema de Negócio

A Associação Passos Mágicos acompanha centenas de estudantes e precisa identificar proativamente aqueles em risco de defasagem escolar para direcionar intervenções pedagógicas.

### Solução Proposta

Modelo de Machine Learning que prediz o **risco de defasagem escolar** em três níveis:

| Nível | Código | Descrição |
|-------|--------|-----------|
| 🟢 BAIXO | 0 | Sem defasagem, indicadores estáveis |
| 🟡 MÉDIO | 1 | Defasagem de 1 ano ou tendência de queda |
| 🔴 ALTO | 2 | Defasagem ≥2 anos ou múltiplos indicadores baixos |

### Stack Tecnológica

| Categoria | Tecnologia |
|-----------|------------|
| Linguagem | Python 3.11+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| API | FastAPI |
| Serialização | joblib |
| Testes | pytest (80%+ cobertura) |
| Container | Docker |
| Monitoramento | Evidently AI |
| Documentação | MkDocs |

## 📁 Estrutura do Projeto

```
passos_magicos_ml/
│
├── data/
│   ├── raw/                          # Dados brutos originais
│   ├── processed/                    # Dados processados
│   └── external/                     # Dados externos
│
├── src/
│   ├── config.py                     # Configurações e constantes
│   ├── preprocessing/                # Pipeline de pré-processamento
│   │   ├── data_loader.py           
│   │   ├── data_cleaner.py          
│   │   └── data_transformer.py      
│   ├── features/                     # Engenharia de features
│   │   ├── feature_engineering.py   
│   │   └── feature_selection.py     
│   ├── models/                       # Modelos de ML
│   │   ├── train.py                 
│   │   ├── evaluate.py              
│   │   └── predict.py               
│   ├── utils/                        # Utilidades
│   │   ├── logger.py                
│   │   └── helpers.py               
│   └── monitoring/                   # Monitoramento
│       ├── drift_detector.py        
│       └── metrics_tracker.py       
│
├── api/                              # API FastAPI
│   ├── main.py                      
│   ├── schemas.py                   
│   ├── routes/                      
│   └── middleware/                  
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_eda.ipynb                 
│   ├── 02_feature_engineering.ipynb 
│   ├── 03_model_selection.ipynb     
│   └── 04_model_evaluation.ipynb    
│
├── tests/                            # Testes pytest
├── models/                           # Modelos serializados
├── logs/                             # Logs da aplicação
├── monitoring/                       # Relatórios de monitoramento
├── docs/                             # Documentação
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.11+
- pip ou conda
- Docker (opcional)

### Setup Local

```bash
# 1. Clonar repositório
git clone https://github.com/mzmvyp/tecchallenge_5.git
cd tecchallenge_5

# 2. Criar ambiente virtual
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instalar dependências
make install

# Ou manualmente:
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Copiar variáveis de ambiente
copy .env.example .env
```

## 💻 Uso

### Treinamento do Modelo

```bash
# Treinar modelo completo
make train

# Ou manualmente
python -m src.models.train
```

### Executar API

```bash
# Modo desenvolvimento (com hot-reload)
make run-dev

# Modo produção
make run
```

A API estará disponível em: http://localhost:8000

### Documentação Interativa

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔌 API

### Endpoints Principais

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Predição Individual
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "bolsista": false
  }'
```

#### Resposta
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

#### Predição em Lote
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"students": [...]}'
```

#### Métricas
```bash
curl http://localhost:8000/monitoring/metrics
```

## 🔬 Pipeline de ML

### Indicadores Utilizados

| Indicador | Descrição |
|-----------|-----------|
| **INDE** | Índice de Desenvolvimento Educacional |
| **IAN** | Indicador de Adequação ao Nível |
| **IDA** | Indicador de Desempenho Acadêmico |
| **IEG** | Indicador de Engajamento |
| **IAA** | Indicador de Autoavaliação |
| **IPS** | Indicador Psicossocial |
| **IPP** | Indicador Psicopedagógico |
| **IPV** | Indicador de Ponto de Virada |

### Features Engenheiradas

- **Defasagem**: Diferença entre fase ideal e atual
- **Temporais**: Anos na PM, veterano
- **Agregações**: Média, desvio padrão, min/max dos indicadores
- **Interações**: Ratios entre indicadores, scores combinados

### Modelos Testados

1. Logistic Regression (baseline)
2. Random Forest
3. Gradient Boosting
4. XGBoost

### Métrica Principal: Recall

**Justificativa**: No contexto educacional, é crítico identificar **todos** os alunos em risco. Um falso negativo (aluno em risco não identificado) tem consequências mais graves que um falso positivo.

## 📊 Monitoramento

### Detecção de Drift

O sistema utiliza Evidently AI para monitorar:
- Drift nas features de entrada
- Drift no target
- Qualidade dos dados

```python
from src.monitoring import DriftDetector

detector = DriftDetector(reference_data=df_train)
results = detector.detect_data_drift(df_production)
```

### Métricas de Produção

- Latência das predições
- Distribuição de classes preditas
- Taxa de erros
- Throughput

## 🧪 Testes

```bash
# Executar todos os testes
make test

# Executar com cobertura
make test-cov

# Verificar cobertura mínima (80%)
pytest tests/ -v --cov=src --cov=api --cov-fail-under=80
```

### Estrutura de Testes

- `test_preprocessing.py` - Testes de pré-processamento
- `test_features.py` - Testes de feature engineering
- `test_models.py` - Testes de modelos
- `test_api.py` - Testes da API
- `test_utils.py` - Testes de utilidades

## 🐳 Docker

### Build e Execução

```bash
# Build da imagem
make docker-build

# Executar com docker-compose
make docker-run

# Parar containers
make docker-stop
```

### Docker Compose

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.joblib
```

## 📚 Documentação Adicional

- [Documentação da API](docs/api_reference.md)
- [Pipeline de ML](docs/pipeline.md)
- [Guia de Deploy](docs/deployment.md)

## 🤝 Contribuição

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

**Desenvolvido para o Datathon POSTECH/FIAP - Passos Mágicos** 🎓✨
