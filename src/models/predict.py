"""
Módulo de predição para o projeto Passos Mágicos.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RISCO_LABELS, settings
from src.utils.logger import get_logger
from src.utils.helpers import load_model

logger = get_logger(__name__)


class ModelPredictor:
    """Classe para realizar predições com o modelo treinado."""
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Inicializa o ModelPredictor.
        
        Args:
            model_path: Caminho para o modelo serializado
        """
        if model_path is None:
            model_path = MODELS_DIR / "model.joblib"
        
        self.model_path = Path(model_path)
        self.model: Optional[Any] = None
        self.pipeline: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        
        self._load_model()
        
        logger.info(f"ModelPredictor inicializado com modelo: {self.model_path}")
    
    def _load_model(self) -> None:
        """Carrega o modelo do disco."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {self.model_path}"
            )
        
        model_data = load_model(self.model_path)
        
        # Extrair componentes
        if isinstance(model_data, dict):
            self.pipeline = model_data.get("pipeline")
            self.model = model_data.get("pipeline")
            self.metadata = model_data.get("metadata", {})
            self.feature_names = model_data.get("feature_names", [])
        else:
            # Modelo direto (sem wrapper)
            self.pipeline = model_data
            self.model = model_data
            self.feature_names = []
        
        logger.info(f"Modelo carregado: {self.metadata}")
        logger.info(f"Features esperadas: {self.feature_names}")
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict, List[Dict]]
    ) -> np.ndarray:
        """
        Realiza predição de classe.
        
        Args:
            X: Features para predição (DataFrame, array, dict ou lista de dicts)
        
        Returns:
            Array com classes preditas
        """
        X_processed = self._preprocess_input(X)
        
        predictions = self.pipeline.predict(X_processed)
        
        logger.debug(f"Predições realizadas: {len(predictions)} registros")
        
        return predictions
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict, List[Dict]]
    ) -> np.ndarray:
        """
        Realiza predição de probabilidades.
        
        Args:
            X: Features para predição
        
        Returns:
            Array com probabilidades por classe
        """
        X_processed = self._preprocess_input(X)
        
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(X_processed)
        else:
            raise ValueError("Modelo não suporta predição de probabilidades")
        
        return probabilities
    
    def predict_single(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Realiza predição para um único estudante.
        
        Args:
            data: Dicionário com dados do estudante
        
        Returns:
            Dicionário com predição e probabilidades
        """
        # Converter para DataFrame
        df = pd.DataFrame([data])
        
        # Predição
        prediction = self.predict(df)[0]
        probabilities = self.predict_proba(df)[0]
        
        # Obter probabilidade da classe predita
        prob_classe = probabilities[int(prediction)]
        
        result = {
            "risco_defasagem": int(prediction),
            "nivel_risco": RISCO_LABELS.get(int(prediction), "DESCONHECIDO"),
            "probabilidade": float(prob_classe),
            "probabilidades_por_classe": {
                RISCO_LABELS.get(i, f"Classe {i}"): float(p)
                for i, p in enumerate(probabilities)
            }
        }
        
        logger.info(
            f"Predição individual: {result['nivel_risco']} "
            f"(prob: {result['probabilidade']:.2%})"
        )
        
        return result
    
    def predict_batch(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Realiza predições em lote.
        
        Args:
            data: DataFrame ou lista de dicionários
        
        Returns:
            Lista de dicionários com predições
        """
        # Converter para DataFrame se necessário
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Predições
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        results = []
        for i in range(len(df)):
            pred = int(predictions[i])
            probs = probabilities[i]
            
            results.append({
                "risco_defasagem": pred,
                "nivel_risco": RISCO_LABELS.get(pred, "DESCONHECIDO"),
                "probabilidade": float(probs[pred]),
                "probabilidades_por_classe": {
                    RISCO_LABELS.get(j, f"Classe {j}"): float(p)
                    for j, p in enumerate(probs)
                }
            })
        
        logger.info(f"Predição em lote: {len(results)} registros processados")
        
        return results
    
    def _preprocess_input(
        self,
        X: Union[pd.DataFrame, np.ndarray, Dict, List[Dict]]
    ) -> np.ndarray:
        """
        Preprocessa entrada para predição.
        
        Args:
            X: Dados de entrada em vários formatos
        
        Returns:
            Array numpy pronto para predição
        """
        # Converter dict ou lista de dicts para DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X = pd.DataFrame(X)
        
        # Converter DataFrame para array
        if isinstance(X, pd.DataFrame):
            # Selecionar apenas colunas numéricas
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols].values
        
        return X
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado.
        
        Returns:
            Dicionário com informações do modelo
        """
        info = {
            "model_path": str(self.model_path),
            "metadata": self.metadata,
            "model_type": type(self.pipeline).__name__,
        }
        
        # Tentar obter informações adicionais do pipeline
        if hasattr(self.pipeline, "named_steps"):
            info["pipeline_steps"] = list(self.pipeline.named_steps.keys())
        
        return info


def predict(
    X: Union[pd.DataFrame, np.ndarray, Dict, List[Dict]],
    model_path: Optional[Path] = None
) -> np.ndarray:
    """
    Função de conveniência para predição.
    
    Args:
        X: Features para predição
        model_path: Caminho do modelo
    
    Returns:
        Array com predições
    """
    predictor = ModelPredictor(model_path)
    return predictor.predict(X)


def predict_student(
    student_data: Dict[str, Any],
    model_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Função de conveniência para predição de um estudante.
    
    Args:
        student_data: Dados do estudante
        model_path: Caminho do modelo
    
    Returns:
        Dicionário com resultado da predição
    """
    predictor = ModelPredictor(model_path)
    return predictor.predict_single(student_data)


class RiskPredictor:
    """
    Classe de alto nível para predição de risco de defasagem.
    Inclui pré-processamento completo dos dados de entrada.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Inicializa o RiskPredictor.
        
        Args:
            model_path: Caminho para o modelo
        """
        self.predictor = ModelPredictor(model_path)
        
        # Colunas esperadas pelo modelo (ordem importa!)
        self._expected_features: List[str] = []
    
    def set_expected_features(self, features: List[str]) -> None:
        """Define as features esperadas pelo modelo."""
        self._expected_features = features
    
    def predict_risk(
        self,
        fase: int,
        idade: int,
        genero: str,
        anos_na_pm: int,
        inde: float,
        ian: float,
        ida: float,
        ieg: float,
        iaa: float,
        ips: float,
        ipp: float,
        ipv: float,
        pedra: str,
        instituicao_ensino: str,
        bolsista: bool,
        ponto_virada: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prediz risco de defasagem para um estudante.
        
        Args:
            fase: Fase atual do estudante na PM
            idade: Idade do estudante
            genero: Gênero (M/F)
            anos_na_pm: Anos na Passos Mágicos
            inde: Índice de Desenvolvimento Educacional
            ian: Indicador de Adequação ao Nível
            ida: Indicador de Desempenho Acadêmico
            ieg: Indicador de Engajamento
            iaa: Indicador de Autoavaliação
            ips: Indicador Psicossocial
            ipp: Indicador Psicopedagógico
            ipv: Indicador de Ponto de Virada
            pedra: Classificação PEDRA
            instituicao_ensino: Instituição de ensino
            bolsista: Se é bolsista
            ponto_virada: Se atingiu ponto de virada
        
        Returns:
            Dicionário com predição completa
        """
        from datetime import datetime
        
        # Montar dados do estudante
        student_data = {
            "FASE": fase,
            "IDADE": idade,
            "ANOS_PM": anos_na_pm,
            "INDE": inde,
            "IAN": ian,
            "IDA": ida,
            "IEG": ieg,
            "IAA": iaa,
            "IPS": ips,
            "IPP": ipp,
            "IPV": ipv,
            "BOLSISTA": int(bolsista),
            "PONTO_VIRADA": int(ponto_virada),
        }
        
        # Calcular features derivadas
        # NOTA: Para evitar data leakage, o modelo NÃO usa:
        # - DEFASAGEM, DEFASAGEM_ABS, TEM_DEFASAGEM
        # - INDICADORES_BAIXOS
        # - SCORE_ACADEMICO, DIFF_ACAD_COMPORT
        # - INDE (diretamente), FASE_IDEAL
        
        # Features de indicadores (sem INDE para evitar data leakage)
        indicadores_sem_inde = [ian, ida, ieg, iaa, ips, ipp, ipv]
        indicadores_com_inde = [inde, ian, ida, ieg, iaa, ips, ipp, ipv]
        
        media_indicadores = np.mean(indicadores_sem_inde)
        std_indicadores = np.std(indicadores_sem_inde) if len(indicadores_sem_inde) > 1 else 0
        cv_indicadores = std_indicadores / media_indicadores if media_indicadores != 0 else 0
        min_indicador = min(indicadores_sem_inde)
        max_indicador = max(indicadores_sem_inde)
        range_indicadores = max_indicador - min_indicador
        indicadores_altos = sum(1 for i in indicadores_sem_inde if i >= 7.0)
        
        # Features de interação (sem SCORE_ACADEMICO para evitar data leakage)
        score_comportamental = np.mean([ieg, ips, ipp, iaa])
        engaj_x_desemp = ieg * ida
        ratio_ida_ieg = ida / ieg if ieg != 0 else 1
        ratio_ipp_ips = ipp / ips if ips != 0 else 1
        inde_x_anospm = inde * (anos_na_pm + 1)
        bolsista_e_virada = int(bolsista and ponto_virada)
        
        # Flag de veterano
        veterano = int(anos_na_pm >= 2)
        
        # Carregar features esperadas do modelo
        # O modelo foi treinado SEM features que causam data leakage
        feature_names = getattr(self.predictor, 'feature_names', None) or [
            # Fallback para modelo sem data leakage
            'FASE', 'IDADE', 'ANOS_PM', 'VETERANO',
            'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV',
            'BOLSISTA', 'PONTO_VIRADA',
            'MEDIA_INDICADORES', 'STD_INDICADORES', 'MIN_INDICADOR', 'MAX_INDICADOR',
            'CV_INDICADORES', 'RANGE_INDICADORES', 'INDICADORES_ALTOS',
            'RATIO_IDA_IEG', 'RATIO_IPP_IPS', 'ENGAJ_X_DESEMP',
            'SCORE_COMPORTAMENTAL', 'INDE_X_ANOSPM', 'BOLSISTA_E_VIRADA'
        ]
        
        # Montar dicionário de todas as features calculadas (sem data leakage)
        all_features = {
            # Dados base
            "FASE": fase,
            "IDADE": idade,
            "ANOS_PM": anos_na_pm,
            "VETERANO": veterano,
            # Indicadores individuais (sem INDE que é usado no target)
            "IAN": ian,
            "IDA": ida,
            "IEG": ieg,
            "IAA": iaa,
            "IPS": ips,
            "IPP": ipp,
            "IPV": ipv,
            # Flags
            "BOLSISTA": int(bolsista),
            "PONTO_VIRADA": int(ponto_virada),
            # Estatísticas dos indicadores
            "MEDIA_INDICADORES": media_indicadores,
            "STD_INDICADORES": std_indicadores,
            "MIN_INDICADOR": min_indicador,
            "MAX_INDICADOR": max_indicador,
            "CV_INDICADORES": cv_indicadores,
            "RANGE_INDICADORES": range_indicadores,
            "INDICADORES_ALTOS": indicadores_altos,
            # Interações
            "RATIO_IDA_IEG": ratio_ida_ieg,
            "RATIO_IPP_IPS": ratio_ipp_ips,
            "ENGAJ_X_DESEMP": engaj_x_desemp,
            "SCORE_COMPORTAMENTAL": score_comportamental,
            "INDE_X_ANOSPM": inde_x_anospm,
            "BOLSISTA_E_VIRADA": bolsista_e_virada,
        }
        
        # Selecionar apenas as features esperadas pelo modelo na ordem correta
        student_data = {k: all_features[k] for k in feature_names if k in all_features}
        
        # Realizar predição
        result = self.predictor.predict_single(student_data)
        
        # Adicionar timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return result
