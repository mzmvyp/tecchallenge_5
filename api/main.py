"""
API FastAPI para predição de risco de defasagem escolar - Passos Mágicos.
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware.logging_middleware import LoggingMiddleware
from api.routes import health_router, predict_router, monitoring_router
from src.config import settings
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager para lifecycle da aplicação.
    
    Executa setup na inicialização e cleanup no shutdown.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Iniciando API Passos Mágicos...")
    logger.info(f"Versão: {settings.model_version}")
    logger.info(f"Ambiente: {settings.environment}")
    logger.info(f"Debug: {settings.debug}")
    logger.info("=" * 60)
    
    # Pré-carregar modelo (opcional)
    try:
        from pathlib import Path
        model_path = Path(settings.model_path)
        if model_path.exists():
            logger.info(f"Modelo encontrado: {model_path}")
        else:
            logger.warning(f"Modelo não encontrado: {model_path}")
            logger.warning("Execute 'make train' para treinar o modelo")
    except Exception as e:
        logger.error(f"Erro ao verificar modelo: {e}")
    
    yield  # Aplicação rodando
    
    # Shutdown
    logger.info("Encerrando API Passos Mágicos...")
    
    # Salvar métricas antes de encerrar
    try:
        from src.monitoring.metrics_tracker import get_metrics_tracker
        tracker = get_metrics_tracker()
        tracker.save_metrics()
        logger.info("Métricas salvas com sucesso")
    except Exception as e:
        logger.error(f"Erro ao salvar métricas: {e}")
    
    logger.info("API encerrada")


# Criar aplicação FastAPI
app = FastAPI(
    title="Passos Mágicos - Predição de Defasagem",
    description="""
    ## API para Predição de Risco de Defasagem Escolar
    
    Esta API faz parte do projeto de Machine Learning da Associação Passos Mágicos,
    uma ONG que transforma a vida de crianças e jovens de baixa renda através da 
    educação em Embu-Guaçu/SP.
    
    ### Funcionalidades
    
    - **Predição Individual**: Avalia o risco de defasagem de um estudante
    - **Predição em Lote**: Processa múltiplos estudantes de uma vez
    - **Monitoramento**: Acompanha métricas e detecta drift nos dados
    - **Health Check**: Verifica status da aplicação
    
    ### Níveis de Risco
    
    | Nível | Descrição |
    |-------|-----------|
    | BAIXO (0) | Estudante com baixo risco de defasagem |
    | MÉDIO (1) | Estudante com risco moderado |
    | ALTO (2) | Estudante com alto risco - requer atenção |
    
    ### Indicadores Utilizados
    
    O modelo utiliza os seguintes indicadores educacionais:
    - **INDE**: Índice de Desenvolvimento Educacional
    - **IAN**: Indicador de Adequação ao Nível
    - **IDA**: Indicador de Desempenho Acadêmico
    - **IEG**: Indicador de Engajamento
    - **IAA**: Indicador de Autoavaliação
    - **IPS**: Indicador Psicossocial
    - **IPP**: Indicador Psicopedagógico
    - **IPV**: Indicador de Ponto de Virada
    
    ---
    
    Desenvolvido para o Datathon POSTECH/FIAP - Passos Mágicos
    """,
    version=settings.model_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens permitidas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adicionar middleware de logging
app.add_middleware(LoggingMiddleware)

# Registrar rotas
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(monitoring_router)


# Handler de exceções global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handler global para exceções não tratadas.
    """
    logger.error(f"Exceção não tratada: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.debug else "Erro interno do servidor",
            "timestamp": datetime.now().isoformat()
        }
    )


# Rota raiz
@app.get("/", tags=["Root"])
async def root():
    """
    Rota raiz da API.
    
    Retorna informações básicas e links úteis.
    """
    return {
        "name": "Passos Mágicos ML API",
        "description": "API para predição de risco de defasagem escolar",
        "version": settings.model_version,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/monitoring/metrics"
    }


# Para execução direta
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
