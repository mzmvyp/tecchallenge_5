"""
Módulo de rastreamento de métricas para o projeto Passos Mágicos.
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.config import MONITORING_DIR, RISCO_LABELS
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)


class MetricsTracker:
    """Classe para rastreamento de métricas de predição e performance."""
    
    def __init__(self, persist: bool = True):
        """
        Inicializa o MetricsTracker.
        
        Args:
            persist: Se deve persistir métricas em disco
        """
        self.persist = persist
        self._metrics_dir = MONITORING_DIR / "metrics"
        
        # Contadores
        self._prediction_counts: Dict[int, int] = defaultdict(int)
        self._latencies: List[float] = []
        self._errors: List[Dict[str, Any]] = []
        self._daily_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Métricas agregadas
        self._total_predictions = 0
        self._total_errors = 0
        self._session_start = datetime.now()
        
        if self.persist:
            ensure_dir(self._metrics_dir)
            self._load_persisted_metrics()
        
        logger.info("MetricsTracker inicializado")
    
    def track_prediction(
        self,
        prediction: int,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registra uma predição.
        
        Args:
            prediction: Classe predita
            latency_ms: Latência em milissegundos
            metadata: Metadados adicionais
        """
        self._total_predictions += 1
        self._prediction_counts[prediction] += 1
        
        # Contagem diária
        today = datetime.now().strftime("%Y-%m-%d")
        self._daily_counts[today][prediction] += 1
        
        # Latência
        if latency_ms is not None:
            self._latencies.append(latency_ms)
        
        logger.debug(
            f"Predição rastreada: {RISCO_LABELS.get(prediction, prediction)} "
            f"(total: {self._total_predictions})"
        )
    
    def track_error(
        self,
        error_type: str,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registra um erro.
        
        Args:
            error_type: Tipo do erro
            error_message: Mensagem do erro
            metadata: Metadados adicionais
        """
        self._total_errors += 1
        
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "metadata": metadata or {}
        }
        
        self._errors.append(error_record)
        
        # Manter apenas últimos 1000 erros
        if len(self._errors) > 1000:
            self._errors = self._errors[-1000:]
        
        logger.warning(f"Erro rastreado: {error_type} - {error_message}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo das métricas.
        
        Returns:
            Dicionário com métricas agregadas
        """
        # Calcular estatísticas de latência
        latency_stats = {}
        if self._latencies:
            latency_stats = {
                "mean_ms": np.mean(self._latencies),
                "median_ms": np.median(self._latencies),
                "p95_ms": np.percentile(self._latencies, 95),
                "p99_ms": np.percentile(self._latencies, 99),
                "min_ms": min(self._latencies),
                "max_ms": max(self._latencies),
            }
        
        # Distribuição de predições
        prediction_distribution = {
            RISCO_LABELS.get(k, f"Classe {k}"): v
            for k, v in self._prediction_counts.items()
        }
        
        # Calcular taxas
        uptime = (datetime.now() - self._session_start).total_seconds()
        predictions_per_minute = (
            self._total_predictions / (uptime / 60) if uptime > 0 else 0
        )
        error_rate = (
            self._total_errors / self._total_predictions 
            if self._total_predictions > 0 else 0
        )
        
        return {
            "total_predictions": self._total_predictions,
            "total_errors": self._total_errors,
            "error_rate": error_rate,
            "prediction_distribution": prediction_distribution,
            "latency": latency_stats,
            "predictions_per_minute": predictions_per_minute,
            "session_uptime_seconds": uptime,
            "session_start": self._session_start.isoformat(),
        }
    
    def get_daily_metrics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retorna métricas de um dia específico.
        
        Args:
            date: Data no formato YYYY-MM-DD (None = hoje)
        
        Returns:
            Dicionário com métricas do dia
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        daily_counts = dict(self._daily_counts.get(date, {}))
        
        return {
            "date": date,
            "total_predictions": sum(daily_counts.values()),
            "distribution": {
                RISCO_LABELS.get(k, f"Classe {k}"): v
                for k, v in daily_counts.items()
            }
        }
    
    def get_recent_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna erros mais recentes.
        
        Args:
            n: Número de erros a retornar
        
        Returns:
            Lista de erros
        """
        return self._errors[-n:]
    
    def get_latency_histogram(self, bins: int = 20) -> Dict[str, Any]:
        """
        Retorna histograma de latências.
        
        Args:
            bins: Número de bins
        
        Returns:
            Dicionário com dados do histograma
        """
        if not self._latencies:
            return {"bins": [], "counts": [], "edges": []}
        
        counts, edges = np.histogram(self._latencies, bins=bins)
        
        return {
            "counts": counts.tolist(),
            "edges": edges.tolist(),
            "bin_labels": [
                f"{edges[i]:.1f}-{edges[i+1]:.1f}"
                for i in range(len(edges) - 1)
            ]
        }
    
    def save_metrics(self) -> Path:
        """
        Salva métricas em disco.
        
        Returns:
            Caminho do arquivo salvo
        """
        if not self.persist:
            logger.warning("Persistência desabilitada")
            return None
        
        metrics = self.get_metrics_summary()
        metrics["daily_counts"] = dict(self._daily_counts)
        metrics["errors"] = self._errors[-100:]  # Últimos 100 erros
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self._metrics_dir / f"metrics_{timestamp}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métricas salvas em: {filepath}")
        
        return filepath
    
    def _load_persisted_metrics(self) -> None:
        """Carrega métricas persistidas anteriormente."""
        try:
            # Encontrar arquivo mais recente
            metric_files = sorted(
                self._metrics_dir.glob("metrics_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not metric_files:
                return
            
            latest = metric_files[0]
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Restaurar contadores diários
            for date, counts in data.get("daily_counts", {}).items():
                for pred, count in counts.items():
                    self._daily_counts[date][int(pred)] = count
            
            logger.info(f"Métricas carregadas de: {latest}")
        
        except Exception as e:
            logger.warning(f"Erro ao carregar métricas persistidas: {e}")
    
    def reset(self) -> None:
        """Reseta todas as métricas."""
        self._prediction_counts.clear()
        self._latencies.clear()
        self._errors.clear()
        self._daily_counts.clear()
        self._total_predictions = 0
        self._total_errors = 0
        self._session_start = datetime.now()
        
        logger.info("Métricas resetadas")


class LatencyTracker:
    """Context manager para rastrear latência de operações."""
    
    def __init__(
        self,
        metrics_tracker: MetricsTracker,
        operation_name: str = "prediction"
    ):
        """
        Inicializa o LatencyTracker.
        
        Args:
            metrics_tracker: Instância do MetricsTracker
            operation_name: Nome da operação
        """
        self.metrics_tracker = metrics_tracker
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.latency_ms: Optional[float] = None
    
    def __enter__(self) -> "LatencyTracker":
        """Início da medição."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Fim da medição."""
        end_time = time.perf_counter()
        self.latency_ms = (end_time - self.start_time) * 1000
        
        if exc_type is not None:
            self.metrics_tracker.track_error(
                error_type=str(exc_type.__name__),
                error_message=str(exc_val),
                metadata={"operation": self.operation_name}
            )
            return False
        
        return True


# Instância global para uso compartilhado
_global_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """
    Retorna instância global do MetricsTracker.
    
    Returns:
        MetricsTracker singleton
    """
    global _global_tracker
    
    if _global_tracker is None:
        _global_tracker = MetricsTracker()
    
    return _global_tracker


def track_prediction(
    prediction: int,
    latency_ms: Optional[float] = None
) -> None:
    """
    Função de conveniência para rastrear predição.
    
    Args:
        prediction: Classe predita
        latency_ms: Latência em ms
    """
    get_metrics_tracker().track_prediction(prediction, latency_ms)


def get_metrics() -> Dict[str, Any]:
    """
    Função de conveniência para obter métricas.
    
    Returns:
        Dicionário com métricas
    """
    return get_metrics_tracker().get_metrics_summary()
