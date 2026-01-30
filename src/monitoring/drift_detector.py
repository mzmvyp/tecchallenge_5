"""
Módulo de detecção de drift para o projeto Passos Mágicos.
Utiliza Evidently AI para monitoramento de dados e modelos.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config import MONITORING_DIR
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)

# Tentar importar Evidently
try:
    from evidently import ColumnMapping
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
    )
    from evidently.report import Report
    from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
    from evidently.test_suite import TestSuite
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently AI não disponível. Instale com: pip install evidently")


class DriftDetector:
    """Classe para detecção de drift em dados e modelos."""
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        target_column: str = "RISCO_DEFASAGEM",
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ):
        """
        Inicializa o DriftDetector.
        
        Args:
            reference_data: Dados de referência (treino)
            target_column: Nome da coluna target
            numerical_features: Lista de features numéricas
            categorical_features: Lista de features categóricas
        """
        self.reference_data = reference_data
        self.target_column = target_column
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        
        self._column_mapping: Optional[Any] = None
        self._reports_dir = MONITORING_DIR / "drift_reports"
        
        ensure_dir(self._reports_dir)
        
        logger.info(
            f"DriftDetector inicializado. "
            f"Evidently disponível: {EVIDENTLY_AVAILABLE}"
        )
    
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """
        Define dados de referência para comparação.
        
        Args:
            data: DataFrame com dados de referência
        """
        self.reference_data = data.copy()
        self._infer_features()
        logger.info(f"Dados de referência definidos: {len(data)} registros")
    
    def _infer_features(self) -> None:
        """Infere features numéricas e categóricas automaticamente."""
        if self.reference_data is None:
            return
        
        if not self.numerical_features:
            self.numerical_features = list(
                self.reference_data.select_dtypes(include=[np.number]).columns
            )
            # Remover target se presente
            if self.target_column in self.numerical_features:
                self.numerical_features.remove(self.target_column)
        
        if not self.categorical_features:
            self.categorical_features = list(
                self.reference_data.select_dtypes(include=["object", "category"]).columns
            )
    
    def _get_column_mapping(self) -> Any:
        """Retorna mapeamento de colunas para Evidently."""
        if not EVIDENTLY_AVAILABLE:
            return None
        
        return ColumnMapping(
            target=self.target_column if self.target_column in self.reference_data.columns else None,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Detecta drift nos dados.
        
        Args:
            current_data: Dados atuais para comparar
            save_report: Se deve salvar relatório HTML
        
        Returns:
            Dicionário com resultados do drift
        """
        if self.reference_data is None:
            raise ValueError("Dados de referência não definidos. Use set_reference_data()")
        
        logger.info("Detectando data drift...")
        
        # Se Evidently não disponível, usar análise simplificada
        if not EVIDENTLY_AVAILABLE:
            return self._simple_drift_analysis(current_data)
        
        # Criar relatório de drift
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        column_mapping = self._get_column_mapping()
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Salvar relatório HTML
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self._reports_dir / f"data_drift_{timestamp}.html"
            report.save_html(str(report_path))
            logger.info(f"Relatório salvo em: {report_path}")
        
        # Extrair resultados
        results = self._extract_drift_results(report)
        
        return results
    
    def _simple_drift_analysis(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Análise simplificada de drift quando Evidently não está disponível.
        
        Args:
            current_data: Dados atuais
        
        Returns:
            Dicionário com análise de drift
        """
        results = {
            "drift_detected": False,
            "features_with_drift": [],
            "summary": {}
        }
        
        for col in self.numerical_features:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue
            
            ref_mean = self.reference_data[col].mean()
            ref_std = self.reference_data[col].std()
            curr_mean = current_data[col].mean()
            
            # Detectar drift se média mudou mais de 2 desvios padrão
            if ref_std > 0:
                z_score = abs(curr_mean - ref_mean) / ref_std
                if z_score > 2:
                    results["drift_detected"] = True
                    results["features_with_drift"].append({
                        "feature": col,
                        "z_score": z_score,
                        "ref_mean": ref_mean,
                        "curr_mean": curr_mean
                    })
            
            results["summary"][col] = {
                "reference_mean": ref_mean,
                "current_mean": curr_mean,
                "reference_std": ref_std
            }
        
        logger.info(
            f"Análise de drift concluída. "
            f"Drift detectado: {results['drift_detected']}"
        )
        
        return results
    
    def _extract_drift_results(self, report: Any) -> Dict[str, Any]:
        """Extrai resultados do relatório Evidently."""
        results = {
            "drift_detected": False,
            "features_with_drift": [],
            "overall_drift_score": 0.0,
        }
        
        try:
            # Extrair métricas do relatório
            report_dict = report.as_dict()
            
            for metric in report_dict.get("metrics", []):
                result = metric.get("result", {})
                
                # Data drift
                if "drift_share" in result:
                    results["overall_drift_score"] = result["drift_share"]
                    results["drift_detected"] = result["drift_share"] > 0.5
                
                # Drift por feature
                if "drift_by_columns" in result:
                    for col, col_result in result["drift_by_columns"].items():
                        if col_result.get("drift_detected", False):
                            results["features_with_drift"].append({
                                "feature": col,
                                "drift_score": col_result.get("drift_score", 0),
                                "stattest": col_result.get("stattest_name", "unknown")
                            })
        
        except Exception as e:
            logger.error(f"Erro ao extrair resultados do drift: {e}")
        
        return results
    
    def run_data_quality_tests(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Executa testes de qualidade de dados.
        
        Args:
            current_data: Dados a testar
        
        Returns:
            Dicionário com resultados dos testes
        """
        if not EVIDENTLY_AVAILABLE:
            return self._simple_quality_check(current_data)
        
        logger.info("Executando testes de qualidade de dados...")
        
        test_suite = TestSuite(tests=[
            DataQualityTestPreset(),
            DataDriftTestPreset(),
        ])
        
        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self._get_column_mapping()
        )
        
        # Salvar relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self._reports_dir / f"quality_tests_{timestamp}.html"
        test_suite.save_html(str(report_path))
        
        # Extrair resultados
        results = test_suite.as_dict()
        
        return {
            "passed": all(t.get("status") == "SUCCESS" for t in results.get("tests", [])),
            "total_tests": len(results.get("tests", [])),
            "failed_tests": [
                t for t in results.get("tests", []) 
                if t.get("status") != "SUCCESS"
            ],
            "report_path": str(report_path)
        }
    
    def _simple_quality_check(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Verificação simplificada de qualidade de dados.
        
        Args:
            data: DataFrame a verificar
        
        Returns:
            Dicionário com resultados
        """
        issues = []
        
        # Verificar valores nulos
        null_pct = data.isnull().mean()
        high_null_cols = null_pct[null_pct > 0.1].index.tolist()
        if high_null_cols:
            issues.append({
                "type": "high_null_rate",
                "columns": high_null_cols,
                "severity": "warning"
            })
        
        # Verificar duplicatas
        dup_pct = data.duplicated().mean()
        if dup_pct > 0.05:
            issues.append({
                "type": "high_duplicate_rate",
                "rate": dup_pct,
                "severity": "warning"
            })
        
        # Verificar ranges dos indicadores
        indicator_cols = [c for c in data.columns if c.startswith(("IND", "IAN", "IDA", "IEG"))]
        for col in indicator_cols:
            if col in data.columns:
                out_of_range = ((data[col] < 0) | (data[col] > 10)).mean()
                if out_of_range > 0:
                    issues.append({
                        "type": "out_of_range",
                        "column": col,
                        "rate": out_of_range,
                        "severity": "error"
                    })
        
        return {
            "passed": len([i for i in issues if i["severity"] == "error"]) == 0,
            "issues": issues,
            "total_issues": len(issues)
        }
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo de todos os relatórios de drift.
        
        Returns:
            Dicionário com resumo
        """
        reports = list(self._reports_dir.glob("*.html"))
        
        return {
            "total_reports": len(reports),
            "reports_dir": str(self._reports_dir),
            "latest_reports": [
                {"name": r.name, "date": r.stat().st_mtime}
                for r in sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            ]
        }


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str = "RISCO_DEFASAGEM"
) -> Dict[str, Any]:
    """
    Função de conveniência para detectar drift.
    
    Args:
        reference_data: Dados de referência
        current_data: Dados atuais
        target_column: Coluna target
    
    Returns:
        Dicionário com resultados do drift
    """
    detector = DriftDetector(
        reference_data=reference_data,
        target_column=target_column
    )
    
    return detector.detect_data_drift(current_data)
