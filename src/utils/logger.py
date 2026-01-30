"""
Sistema de logging para o projeto Passos Mágicos ML.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import settings, LOGS_DIR


def setup_logger(
    name: str = "passos_magicos",
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configura e retorna um logger.
    
    Args:
        name: Nome do logger
        log_file: Caminho para o arquivo de log (opcional)
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Formato das mensagens de log
    
    Returns:
        Logger configurado
    """
    # Usar configurações padrão se não fornecidas
    log_level = level or settings.log_level
    log_file = log_file or settings.log_file
    
    # Formato padrão
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
    
    # Criar logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Evitar duplicação de handlers
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (se especificado)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "passos_magicos") -> logging.Logger:
    """
    Retorna um logger existente ou cria um novo.
    
    Args:
        name: Nome do logger
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Se o logger não tem handlers, configura
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerContext:
    """Context manager para logging de operações."""
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        level: int = logging.INFO
    ):
        """
        Inicializa o context manager.
        
        Args:
            logger: Logger a ser usado
            operation: Nome da operação sendo executada
            level: Nível de log
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time: Optional[datetime] = None
    
    def __enter__(self):
        """Início da operação."""
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Iniciando: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fim da operação."""
        duration = datetime.now() - self.start_time
        
        if exc_type is not None:
            self.logger.error(
                f"Erro em '{self.operation}': {exc_val}",
                exc_info=True
            )
            return False
        
        self.logger.log(
            self.level,
            f"Concluído: {self.operation} (duração: {duration.total_seconds():.2f}s)"
        )
        return True


# Logger global para uso rápido
logger = setup_logger()
