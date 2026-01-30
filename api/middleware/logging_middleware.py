"""
Middleware de logging para a API Passos Mágicos.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger("api.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requisições e respostas."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Processa requisição e loga informações.
        
        Args:
            request: Requisição HTTP
            call_next: Próximo middleware/handler
        
        Returns:
            Resposta HTTP
        """
        # Gerar ID único para a requisição
        request_id = str(uuid.uuid4())[:8]
        
        # Timestamp de início
        start_time = time.perf_counter()
        
        # Informações da requisição
        method = request.method
        url = str(request.url)
        client_ip = request.client.host if request.client else "unknown"
        
        # Log da requisição
        logger.info(
            f"[{request_id}] Requisição: {method} {url} | "
            f"Client: {client_ip}"
        )
        
        # Processar requisição
        try:
            response = await call_next(request)
            
            # Calcular duração
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log da resposta
            logger.info(
                f"[{request_id}] Resposta: {response.status_code} | "
                f"Duração: {duration_ms:.2f}ms"
            )
            
            # Adicionar headers de rastreamento
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
        
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            logger.error(
                f"[{request_id}] Erro: {type(e).__name__}: {str(e)} | "
                f"Duração: {duration_ms:.2f}ms"
            )
            
            raise


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware para validação de requisições."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Valida requisição antes de processar.
        
        Args:
            request: Requisição HTTP
            call_next: Próximo middleware/handler
        
        Returns:
            Resposta HTTP
        """
        # Verificar Content-Type para requisições POST/PUT
        if request.method in ["POST", "PUT"]:
            content_type = request.headers.get("content-type", "")
            
            # Permitir apenas JSON
            if content_type and "application/json" not in content_type:
                logger.warning(
                    f"Content-Type inválido: {content_type}. "
                    f"Esperado: application/json"
                )
        
        return await call_next(request)
