# Pipeline de Machine Learning

## Visão Geral do Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dados     │ -> │   Limpeza   │ -> │  Features   │ -> │  Modelo     │
│   Brutos    │    │   e Prep    │    │ Engineering │    │  Training   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
                                                               v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Deploy    │ <- │   API       │ <- │ Serialização│ <- │  Avaliação  │
│             │    │   FastAPI   │    │   joblib    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 1. Pré-processamento

### DataLoader

Carrega dados de múltiplas sheets do Excel e unifica em um único DataFrame.

```python
from src.preprocessing import DataLoader

loader = DataLoader()
df = loader.load_and_unify()
```

### DataCleaner

- Tratamento de valores missing (estratégias: median, mean, mode)
- Remoção de duplicatas
- Validação de ranges (indicadores 0-10)
- Padronização de tipos

```python
from src.preprocessing import DataCleaner

cleaner = DataCleaner(numeric_strategy="median")
df_clean = cleaner.fit_transform(df)
```

### DataTransformer

- Normalização/Padronização de features numéricas
- Encoding de variáveis categóricas (OneHot, Label)

```python
from src.preprocessing import DataTransformer

transformer = DataTransformer(scaling_method="standard")
df_transformed = transformer.fit_transform(df)
```

## 2. Feature Engineering

### Features Criadas

| Feature | Descrição | Fórmula |
|---------|-----------|---------|
| FASE_IDEAL | Fase esperada pela idade | IDADE - 6 |
| DEFASAGEM | Diferença entre ideal e atual | FASE_IDEAL - FASE |
| MEDIA_INDICADORES | Média de todos indicadores | mean(IND*) |
| STD_INDICADORES | Variabilidade dos indicadores | std(IND*) |
| INDICADORES_BAIXOS | Contagem de indicadores < 5.5 | count(IND < 5.5) |
| RATIO_IDA_IEG | Desempenho vs Engajamento | IDA / IEG |
| SCORE_ACADEMICO | Score acadêmico combinado | mean(IDA, IAN) |
| SCORE_COMPORTAMENTAL | Score comportamental | mean(IEG, IPS, IPP) |

### Variável Target

```python
def calcular_risco_defasagem(row):
    # ALTO RISCO (2)
    if defasagem >= 2 or inde < 5.5 or indicadores_baixos >= 4:
        return 2
    
    # MÉDIO RISCO (1)
    if defasagem == 1 or (5.5 <= inde < 6.5):
        return 1
    
    # BAIXO RISCO (0)
    return 0
```

## 3. Seleção de Features

### Métodos Disponíveis

1. **Importância (Random Forest)**: Feature importance via árvores
2. **RFE**: Recursive Feature Elimination
3. **SelectKBest**: Seleção estatística (f_classif)

### Filtros Aplicados

- Remoção de baixa variância (threshold: 0.01)
- Remoção de alta correlação (threshold: 0.95)

## 4. Treinamento

### Modelos Suportados

| Modelo | Classe | Configuração |
|--------|--------|--------------|
| Logistic Regression | LogisticRegression | C, penalty, class_weight |
| Random Forest | RandomForestClassifier | n_estimators, max_depth |
| Gradient Boosting | GradientBoostingClassifier | n_estimators, learning_rate |
| XGBoost | XGBClassifier | n_estimators, max_depth |

### Otimização de Hiperparâmetros

- GridSearchCV com validação cruzada estratificada
- 5 folds por padrão
- Métrica de otimização: Recall

### Balanceamento de Classes

- SMOTE (Synthetic Minority Over-sampling Technique)
- class_weight="balanced"

## 5. Avaliação

### Métricas Calculadas

- **Recall** (principal): Não perder alunos em risco
- Precision
- F1-Score
- AUC-ROC
- Matriz de Confusão

### Justificativa do Recall

> No contexto educacional, um falso negativo (aluno em risco não identificado) 
> tem consequências mais graves que um falso positivo. Por isso, priorizamos 
> Recall alto (>85%) mesmo com alguma redução em Precision.

## 6. Serialização e Deploy

### Formato de Serialização

```python
model_data = {
    "pipeline": sklearn_pipeline,
    "model_type": "random_forest",
    "best_params": {...},
    "cv_results": {...},
    "metadata": {
        "version": "1.0.0",
        "scoring": "recall"
    }
}

joblib.dump(model_data, "models/model.joblib")
```

### API de Predição

A API FastAPI carrega o modelo e expõe endpoints para predição individual e em lote.
