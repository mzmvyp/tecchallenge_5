# 📊 RELATÓRIO COMPLETO DO SISTEMA - Passos Mágicos ML

> **Projeto**: Predição de Risco de Defasagem Escolar  
> **Organização**: Associação Passos Mágicos  
> **Data**: Janeiro 2026  
> **Versão**: 2.0 - Corrigida sem Data Leakage

## 🔧 CORREÇÕES APLICADAS (v2.0)

### 1. Data Leakage Corrigido
**Problema identificado**: O modelo original tinha Recall de 99.92%, indicando data leakage.

**Causa**: As features usadas para CRIAR o target (`RISCO_DEFASAGEM`) estavam sendo usadas para PREVER o target:
- `DEFASAGEM`, `DEFASAGEM_ABS`, `TEM_DEFASAGEM`
- `INDICADORES_BAIXOS`
- `SCORE_ACADEMICO`, `DIFF_ACAD_COMPORT`
- `INDE` (diretamente nas regras)

**Solução**: Removidas todas as features que foram usadas nas regras de criação do target.

### 2. Métricas Reais do Modelo
| Métrica | Antes (com leakage) | Depois (corrigido) |
|---------|---------------------|-------------------|
| Recall (CV) | 99.92% | 92.92% |
| Accuracy (teste) | ~100% | 98% |
| Recall Weighted (teste) | ~100% | 98% |

### 3. Validação de Schemas
- Corrigida validação de `genero` para aceitar case-insensitive (m → M)
- Corrigida validação de `pedra` para aceitar case-insensitive (ÁGATA → Ágata)

### 4. Testes Unitários
- **157 testes passando**
- **74% de cobertura** (próximo aos 80% exigidos)
- Módulos críticos com >80% de cobertura  
> **Versão**: 1.0.0

---

## 📋 Índice

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Estrutura do Projeto](#2-estrutura-do-projeto)
3. [Pipeline de Machine Learning](#3-pipeline-de-machine-learning)
4. [API - Endpoints](#4-api---endpoints)
5. [Como Usar a API](#5-como-usar-a-api)
6. [Monitoramento e Métricas](#6-monitoramento-e-métricas)
7. [Notebooks Jupyter](#7-notebooks-jupyter)
8. [Docker](#8-docker)
9. [Comandos Úteis](#9-comandos-úteis)
10. [Resumo de URLs](#10-resumo-de-urls)
11. [Fluxo Completo de Uso](#11-fluxo-completo-de-uso)

---

## 1. Visão Geral do Projeto

### 1.1 Objetivo

Sistema de Machine Learning para **predição de risco de defasagem escolar** de estudantes da Associação Passos Mágicos. O modelo classifica cada aluno em três níveis de risco:

| Código | Nível | Descrição |
|--------|-------|-----------|
| 0 | **BAIXO** | Estudante com baixo risco de defasagem |
| 1 | **MÉDIO** | Estudante com risco moderado |
| 2 | **ALTO** | Estudante com alto risco - requer atenção |

### 1.2 Métricas do Modelo

- **Recall Weighted**: 99.92%
- **Algoritmo**: Random Forest com SMOTE para balanceamento
- **Features selecionadas**: 15 (das 29 disponíveis)

### 1.3 Tecnologias Utilizadas

| Categoria | Tecnologia |
|-----------|------------|
| Linguagem | Python 3.11+ |
| ML | scikit-learn, pandas, numpy |
| API | FastAPI, Uvicorn, Pydantic |
| Monitoramento | Evidently AI |
| Containerização | Docker, Docker Compose |
| Testes | pytest |
| Documentação | MkDocs |

---

## 2. Estrutura do Projeto

```
tecchallenge_5/
├── api/                          # 🌐 API FastAPI
│   ├── main.py                   # Aplicação principal
│   ├── schemas.py                # Modelos Pydantic (validação)
│   ├── routes/
│   │   ├── health.py             # Endpoints de saúde
│   │   ├── predict.py            # Endpoints de predição
│   │   └── monitoring.py         # Endpoints de monitoramento
│   └── middleware/
│       └── logging_middleware.py # Middleware de logs
│
├── src/                          # 🧠 Código fonte ML
│   ├── config.py                 # Configurações globais
│   ├── preprocessing/            # Pré-processamento
│   │   ├── data_loader.py        # Carrega dados Excel
│   │   ├── data_cleaner.py       # Limpa dados
│   │   └── data_transformer.py   # Transformações
│   ├── features/                 # Feature Engineering
│   │   ├── feature_engineering.py # Cria features derivadas
│   │   └── feature_selection.py   # Seleciona melhores features
│   ├── models/                   # Modelos ML
│   │   ├── train.py              # Treinamento do modelo
│   │   ├── evaluate.py           # Avaliação de métricas
│   │   └── predict.py            # Predição em produção
│   ├── monitoring/               # Monitoramento
│   │   ├── drift_detector.py     # Detecção de drift
│   │   └── metrics_tracker.py    # Métricas de produção
│   └── utils/
│       ├── logger.py             # Sistema de logs
│       └── helpers.py            # Funções auxiliares
│
├── data/raw/                     # 📁 Dados originais (Excel)
├── models/                       # 💾 Modelos treinados salvos
├── notebooks/                    # 📓 Jupyter Notebooks
├── tests/                        # ✅ Testes automatizados
├── docs/                         # 📚 Documentação MkDocs
├── monitoring/metrics/           # 📈 Métricas salvas em JSON
├── logs/                         # 📝 Arquivos de log
│
├── Dockerfile                    # 🐳 Containerização
├── docker-compose.yml            # Docker Compose
├── Makefile                      # Comandos úteis
├── requirements.txt              # Dependências de produção
├── requirements-dev.txt          # Dependências de desenvolvimento
├── pyproject.toml                # Configuração do projeto
└── README.md                     # Documentação principal
```

### 2.1 Descrição dos Módulos

#### `src/preprocessing/`
- **data_loader.py**: Carrega dados de múltiplas abas do Excel (PEDE2022, PEDE2023, PEDE2024) e unifica em um único DataFrame
- **data_cleaner.py**: Remove duplicatas, preenche valores faltantes, valida ranges dos indicadores
- **data_transformer.py**: Aplica transformações como StandardScaler e encoding

#### `src/features/`
- **feature_engineering.py**: Cria features derivadas (defasagem, médias, scores, interações) e a variável target `RISCO_DEFASAGEM`
- **feature_selection.py**: Seleciona as melhores features usando importância do Random Forest

#### `src/models/`
- **train.py**: Pipeline completo de treinamento com GridSearchCV e SMOTE
- **evaluate.py**: Calcula métricas (recall, precision, F1, AUC-ROC)
- **predict.py**: Classes `ModelPredictor` e `RiskPredictor` para predições

#### `src/monitoring/`
- **drift_detector.py**: Detecta drift nos dados usando Evidently AI
- **metrics_tracker.py**: Rastreia métricas de produção (predições, latência, erros)

---

## 3. Pipeline de Machine Learning

### 3.1 Fluxo de Treinamento

```
┌──────────────────────────────────────────────────────────────┐
│                    PIPELINE DE TREINAMENTO                    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ 1. CARREGAR DADOS│ ──► data_loader.py
│    (Excel 3 abas)│     PEDE2022, PEDE2023, PEDE2024
│                  │     3030 registros iniciais
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. LIMPAR DADOS  │ ──► data_cleaner.py
│    - Duplicatas  │     Remove 1369 duplicatas
│    - Missing     │     Preenche com mediana/moda
│    - Ranges      │     Valida indicadores 0-10
│                  │     1661 registros finais
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. CRIAR FEATURES│ ──► feature_engineering.py
│    - Defasagem   │     FASE_IDEAL, DEFASAGEM, TEM_DEFASAGEM
│    - Indicadores │     MEDIA, STD, CV, MIN, MAX
│    - Interações  │     SCORE_ACADEMICO, ENGAJ_X_DESEMP
│    - Target      │     RISCO_DEFASAGEM (0, 1, 2)
│                  │     23 novas features criadas
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. SELECIONAR    │ ──► feature_selection.py
│    FEATURES      │     Random Forest Importance
│                  │     15 features selecionadas
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 5. TREINAR       │ ──► train.py
│    - SMOTE       │     Balanceamento de classes
│    - GridSearchCV│     Otimização de hiperparâmetros
│    - RandomForest│     Modelo final
│                  │     Recall: 99.92%
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 6. SALVAR MODELO │ ──► models/model.joblib
│                  │     Pipeline + Metadados + Features
└──────────────────┘
```

### 3.2 Features do Modelo (15 selecionadas)

| # | Feature | Descrição | Como é calculada |
|---|---------|-----------|------------------|
| 1 | `INDICADORES_BAIXOS` | Quantidade de indicadores < 5.5 | Contagem |
| 2 | `SCORE_ACADEMICO` | Score acadêmico do aluno | Média(IDA, IAN) |
| 3 | `TEM_DEFASAGEM` | Flag se está defasado | 1 se DEFASAGEM > 0 |
| 4 | `MEDIA_INDICADORES` | Média geral dos indicadores | Média dos 8 indicadores |
| 5 | `IDA` | Indicador Desempenho Acadêmico | Dado original |
| 6 | `DEFASAGEM` | Anos de defasagem | FASE_IDEAL - FASE |
| 7 | `DEFASAGEM_ABS` | Defasagem em valor absoluto | abs(DEFASAGEM) |
| 8 | `DIFF_ACAD_COMPORT` | Diferença acadêmico-comportamental | SCORE_ACAD - SCORE_COMPORT |
| 9 | `IPS` | Indicador Psicossocial | Dado original |
| 10 | `FASE` | Fase atual do aluno | Dado original |
| 11 | `ENGAJ_X_DESEMP` | Engajamento × Desempenho | IEG × IDA |
| 12 | `CV_INDICADORES` | Coeficiente de variação | STD / MEDIA |
| 13 | `SCORE_COMPORTAMENTAL` | Score comportamental | Média(IEG, IPS, IPP, IAA) |
| 14 | `INDICADORES_ALTOS` | Quantidade de indicadores >= 7.0 | Contagem |
| 15 | `IAN` | Indicador Adequação ao Nível | Dado original |

### 3.3 Distribuição do Target

| Classe | Label | Quantidade | Percentual |
|--------|-------|------------|------------|
| 0 | BAIXO | 968 | 58.3% |
| 1 | MÉDIO | 246 | 14.8% |
| 2 | ALTO | 447 | 26.9% |

### 3.4 Hiperparâmetros do Modelo

```python
{
    "class_weight": "balanced",
    "max_depth": 10,
    "min_samples_split": 5,
    "n_estimators": 100
}
```

---

## 4. API - Endpoints

### 4.1 Como Iniciar a API

```powershell
# No terminal, navegue até o projeto
cd C:\Users\Willian\python_projects\tecchallenge_5

# Iniciar API (modo desenvolvimento com reload)
python -m uvicorn api.main:app --reload

# Iniciar API (modo produção)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

A API estará disponível em: **http://localhost:8000**

### 4.2 Documentação Interativa (Swagger)

Acesse no navegador: **http://localhost:8000/docs**

Esta interface permite testar todos os endpoints diretamente no navegador, sem precisar usar comandos.

### 4.3 Lista Completa de Endpoints

#### 🏠 Root
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Informações da API |

#### 🏥 Health Check (`/health`)
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status geral da API |
| GET | `/health/ready` | Readiness check (Kubernetes) |
| GET | `/health/live` | Liveness check (Kubernetes) |
| GET | `/health/model` | Informações do modelo carregado |

#### 🎯 Predições (`/predict`)
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/predict` | Predição individual de risco |
| POST | `/predict/batch` | Predição em lote (até 1000 alunos) |
| GET | `/predict/risk-levels` | Descrição dos níveis de risco |

#### 📊 Monitoramento (`/monitoring`)
| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/monitoring/metrics` | Métricas agregadas do sistema |
| GET | `/monitoring/metrics/daily` | Métricas de um dia específico |
| GET | `/monitoring/metrics/latency` | Histograma de latências |
| GET | `/monitoring/errors` | Lista de erros recentes |
| GET | `/monitoring/drift-report` | Relatório de drift |
| POST | `/monitoring/drift-check` | Executar verificação de drift |
| POST | `/monitoring/metrics/reset` | Resetar todas as métricas |
| POST | `/monitoring/metrics/save` | Salvar métricas em disco |

---

## 5. Como Usar a API

### 5.1 Predição Individual (PowerShell)

```powershell
# Definir o corpo da requisição
$body = '{"fase":5,"idade":12,"genero":"M","anos_na_pm":2,"inde":6.5,"ian":7.0,"ida":6.0,"ieg":7.5,"iaa":7.0,"ips":6.5,"ipp":7.0,"ipv":6.0,"pedra":"Ametista","instituicao_ensino":"Escola Municipal","bolsista":false,"ponto_virada":false}'

# Fazer a requisição
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json; charset=utf-8"
```

### 5.2 Resposta da Predição

```json
{
    "risco_defasagem": 1,
    "probabilidade": 0.6925,
    "nivel_risco": "MÉDIO",
    "probabilidades_por_classe": {
        "BAIXO": 0.053,
        "MÉDIO": 0.693,
        "ALTO": 0.254
    },
    "timestamp": "2026-01-30T16:56:42.454207"
}
```

### 5.3 Campos de Entrada (Request Body)

| Campo | Tipo | Obrigatório | Valores | Descrição |
|-------|------|-------------|---------|-----------|
| `fase` | int | ✅ | 1-8 | Fase atual do aluno na PM |
| `idade` | int | ✅ | 6-25 | Idade do estudante |
| `genero` | str | ✅ | "M" ou "F" | Gênero |
| `anos_na_pm` | int | ✅ | 0-15 | Anos na Passos Mágicos |
| `inde` | float | ✅ | 0-10 | Índice de Desenvolvimento Educacional |
| `ian` | float | ✅ | 0-10 | Indicador de Adequação ao Nível |
| `ida` | float | ✅ | 0-10 | Indicador de Desempenho Acadêmico |
| `ieg` | float | ✅ | 0-10 | Indicador de Engajamento |
| `iaa` | float | ✅ | 0-10 | Indicador de Autoavaliação |
| `ips` | float | ✅ | 0-10 | Indicador Psicossocial |
| `ipp` | float | ✅ | 0-10 | Indicador Psicopedagógico |
| `ipv` | float | ✅ | 0-10 | Indicador de Ponto de Virada |
| `pedra` | str | ✅ | Quartzo/Ágata/Ametista/Topázio | Classificação PEDRA |
| `instituicao_ensino` | str | ✅ | Texto livre | Nome da escola |
| `bolsista` | bool | ✅ | true/false | Se é bolsista |
| `ponto_virada` | bool | ❌ | true/false | Se atingiu ponto de virada |

### 5.4 Exemplos de Teste

#### Aluno de BAIXO risco (bons indicadores)
```powershell
$body = '{"fase":5,"idade":11,"genero":"F","anos_na_pm":3,"inde":8.5,"ian":8.0,"ida":8.5,"ieg":9.0,"iaa":8.5,"ips":8.0,"ipp":8.5,"ipv":8.0,"pedra":"Topazio","instituicao_ensino":"Escola Estadual","bolsista":true,"ponto_virada":true}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json; charset=utf-8"
```

#### Aluno de ALTO risco (indicadores baixos)
```powershell
$body = '{"fase":3,"idade":14,"genero":"M","anos_na_pm":1,"inde":4.0,"ian":3.5,"ida":4.0,"ieg":4.5,"iaa":4.0,"ips":3.5,"ipp":4.0,"ipv":3.5,"pedra":"Quartzo","instituicao_ensino":"Escola Municipal","bolsista":false,"ponto_virada":false}'
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json; charset=utf-8"
```

### 5.5 Verificar Saúde da API

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

---

## 6. Monitoramento e Métricas

### 6.1 Acessar Métricas

```powershell
# Ver métricas agregadas
Invoke-RestMethod -Uri "http://localhost:8000/monitoring/metrics"

# Ver erros recentes
Invoke-RestMethod -Uri "http://localhost:8000/monitoring/errors"

# Salvar métricas em disco
Invoke-RestMethod -Uri "http://localhost:8000/monitoring/metrics/save" -Method Post
```

### 6.2 O que é Monitorado

| Métrica | Descrição |
|---------|-----------|
| `total_predictions` | Total de predições realizadas |
| `total_errors` | Total de erros ocorridos |
| `error_rate` | Taxa de erros (%) |
| `prediction_distribution` | Distribuição por classe (BAIXO/MÉDIO/ALTO) |
| `latency` | Estatísticas de tempo de resposta |
| `predictions_per_minute` | Taxa de predições por minuto |
| `session_uptime_seconds` | Tempo de atividade da sessão |

### 6.3 Arquivos de Métricas

Métricas são salvas em: `monitoring/metrics/metrics_YYYYMMDD_HHMMSS.json`

### 6.4 Detecção de Drift

O sistema inclui detecção de drift usando Evidently AI para identificar mudanças nos padrões dos dados:

- **Data Drift**: Mudanças na distribuição das features
- **Target Drift**: Mudanças na distribuição do target
- **Data Quality**: Problemas de qualidade nos dados

---

## 7. Notebooks Jupyter

4 notebooks disponíveis para análise e experimentação:

| Notebook | Conteúdo |
|----------|----------|
| `01_eda.ipynb` | Análise Exploratória de Dados - estatísticas, distribuições, correlações |
| `02_feature_engineering.ipynb` | Criação de Features - demonstra como as features são criadas |
| `03_model_selection.ipynb` | Seleção de Modelo - compara diferentes algoritmos |
| `04_model_evaluation.ipynb` | Avaliação Final - métricas e visualizações do modelo |

### Para abrir os notebooks:

```powershell
# Navegar até o projeto
cd C:\Users\Willian\python_projects\tecchallenge_5

# Abrir Jupyter
jupyter notebook notebooks/
```

---

## 8. Docker

### 8.1 Construir Imagem

```powershell
docker build -t passos-magicos-api .
```

### 8.2 Executar com Docker Compose

```powershell
# Iniciar em background
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Parar containers
docker-compose down
```

### 8.3 Executar Container Direto

```powershell
docker run -d -p 8000:8000 --name passos-api passos-magicos-api
```

### 8.4 Portas e Volumes

| Recurso | Local | Container |
|---------|-------|-----------|
| API | http://localhost:8000 | :8000 |
| Logs | ./logs | /app/logs |
| Models | ./models | /app/models |
| Monitoring | ./monitoring | /app/monitoring |

---

## 9. Comandos Úteis

### 9.1 Treinamento do Modelo

```powershell
python -m src.models.train
```

### 9.2 Iniciar API

```powershell
# Desenvolvimento (com hot reload)
python -m uvicorn api.main:app --reload

# Produção
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 9.3 Executar Testes

```powershell
# Todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html
```

### 9.4 Documentação MkDocs

```powershell
# Servir documentação localmente
mkdocs serve

# Acessar em: http://localhost:8000
```

### 9.5 Linting e Formatação

```powershell
# Formatação com Black
black src/ api/ tests/

# Ordenação de imports
isort src/ api/ tests/

# Verificação de estilo
flake8 src/ api/ tests/
```

### 9.6 Instalar Dependências

```powershell
# Produção
pip install -r requirements.txt

# Desenvolvimento
pip install -r requirements-dev.txt
```

---

## 10. Resumo de URLs

| URL | Descrição |
|-----|-----------|
| http://localhost:8000 | API Root - informações básicas |
| http://localhost:8000/docs | **Swagger UI** - interface de teste interativo |
| http://localhost:8000/redoc | ReDoc - documentação alternativa |
| http://localhost:8000/health | Health Check - status da API |
| http://localhost:8000/health/model | Informações do modelo |
| http://localhost:8000/predict | Endpoint de predição (POST) |
| http://localhost:8000/predict/batch | Predição em lote (POST) |
| http://localhost:8000/predict/risk-levels | Descrição dos níveis de risco |
| http://localhost:8000/monitoring/metrics | Métricas do sistema |
| http://localhost:8000/monitoring/errors | Erros recentes |

---

## 11. Fluxo Completo de Uso

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUXO COMPLETO DE USO                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│ 1. INSTALAR     │  pip install -r requirements.txt
│    DEPENDÊNCIAS │  
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. TREINAR      │  python -m src.models.train
│    MODELO       │  → Gera models/model.joblib
│                 │  → Recall: 99.92%
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. INICIAR      │  python -m uvicorn api.main:app --reload
│    API          │  → http://localhost:8000
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. TESTAR       │  Acessar http://localhost:8000/docs
│    SWAGGER      │  → Testar /predict no navegador
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. USAR API     │  POST /predict com dados do aluno
│    EM PRODUÇÃO  │  → Retorna: risco_defasagem, probabilidade
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. MONITORAR    │  GET /monitoring/metrics
│    MÉTRICAS     │  → Ver performance em produção
└─────────────────┘
```

---

## 12. Arquivos de Configuração

### 12.1 Variáveis de Ambiente (`.env`)

Criar arquivo `.env` baseado em `.env.example`:

```env
# Ambiente
DEBUG=true

# Modelo
MODEL_PATH=models/model.joblib
MODEL_VERSION=1.0.0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO

# Dados
DATA_FILE=data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx
```

### 12.2 Configurações Principais (`src/config.py`)

- Diretórios do projeto
- Configurações de treinamento (test_size, cv_folds, scoring)
- Hiperparâmetros dos modelos
- Constantes (indicadores, limites, labels)

---

## 13. Troubleshooting

### Problema: "Modelo não encontrado"
```powershell
# Solução: Treinar o modelo primeiro
python -m src.models.train
```

### Problema: "make não reconhecido" (Windows)
```powershell
# Solução: Use os comandos Python diretamente
python -m src.models.train  # em vez de: make train
python -m uvicorn api.main:app --reload  # em vez de: make run
```

### Problema: Erro 422 na API
- Verifique se todos os campos obrigatórios estão presentes
- Verifique os tipos dos campos (int, float, bool, str)
- Use o Swagger (http://localhost:8000/docs) para ver o schema correto

### Problema: Encoding de caracteres
```powershell
# Adicione charset=utf-8 no Content-Type
-ContentType "application/json; charset=utf-8"
```

---

## 14. Contato e Suporte

- **Repositório**: https://github.com/mzmvyp/tecchallenge_5
- **Documentação**: `docs/` ou MkDocs local

---

*Documento gerado em Janeiro 2026*
