"""
Módulo de engenharia de features para o projeto Passos Mágicos.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    RISCO_THRESHOLDS,
    RISCO_LABELS,
    INDICADORES,
)
from src.utils.logger import get_logger
from src.utils.helpers import categorize_inde, safe_divide

logger = get_logger(__name__)


class FeatureEngineer:
    """Classe para criação de features derivadas."""
    
    def __init__(self):
        """Inicializa o FeatureEngineer."""
        self._created_features: List[str] = []
        logger.info("FeatureEngineer inicializado")
    
    def _ensure_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante que colunas críticas são numéricas.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com tipos corrigidos
        """
        # Colunas que devem ser numéricas
        numeric_cols = [
            "FASE", "IDADE", "INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV",
            "NOTA_PORT", "NOTA_MAT", "NOTA_ING", "NIVEL_IDEAL", "ANO_PEDE", "ANO_INGRESSO"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Tentar converter para numérico
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.debug(f"Coluna '{col}' convertida para numérico")
        
        # Colunas booleanas
        bool_cols = ["BOLSISTA", "PONTO_VIRADA"]
        for col in bool_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Mapear valores booleanos comuns
                    bool_map = {
                        'SIM': True, 'S': True, 'YES': True, 'Y': True, 'TRUE': True, '1': True,
                        'NÃO': False, 'NAO': False, 'N': False, 'NO': False, 'FALSE': False, '0': False,
                        True: True, False: False, 1: True, 0: False
                    }
                    df[col] = df[col].map(lambda x: bool_map.get(str(x).upper().strip(), False) if pd.notna(x) else False)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria todas as features derivadas.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features adicionais
        """
        logger.info(f"Criando features para {len(df)} registros...")
        
        df_feat = df.copy()
        
        # 0. Converter colunas críticas para tipos numéricos
        df_feat = self._ensure_numeric_columns(df_feat)
        
        # 1. Features de defasagem
        df_feat = self.create_defasagem_features(df_feat)
        
        # 2. Features temporais
        df_feat = self.create_temporal_features(df_feat)
        
        # 3. Features de indicadores
        df_feat = self.create_indicator_features(df_feat)
        
        # 4. Features de interação
        df_feat = self.create_interaction_features(df_feat)
        
        # 5. Target: Risco de Defasagem
        df_feat = self.create_target(df_feat)
        
        logger.info(f"Features criadas: {len(self._created_features)} novas colunas")
        
        return df_feat
    
    def create_defasagem_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features relacionadas à defasagem escolar.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features de defasagem
        """
        df = df.copy()
        
        # Calcular fase ideal baseada na idade (aproximação)
        # Assumindo: Fase 1 = 7-8 anos, Fase 2 = 8-9 anos, etc.
        if "IDADE" in df.columns:
            # Verificar se IDADE é numérica
            if pd.api.types.is_numeric_dtype(df["IDADE"]):
                df["FASE_IDEAL"] = np.clip(df["IDADE"] - 6, 1, 8)
                self._created_features.append("FASE_IDEAL")
            elif pd.api.types.is_datetime64_any_dtype(df["IDADE"]):
                # IDADE é uma data (nascimento), calcular idade a partir dela
                from datetime import datetime
                ano_referencia = df["ANO_PEDE"].iloc[0] if "ANO_PEDE" in df.columns else datetime.now().year
                df["IDADE_CALC"] = ano_referencia - df["IDADE"].dt.year
                df["FASE_IDEAL"] = np.clip(df["IDADE_CALC"] - 6, 1, 8)
                self._created_features.extend(["IDADE_CALC", "FASE_IDEAL"])
            else:
                # Tentar converter para numérico
                try:
                    df["IDADE"] = pd.to_numeric(df["IDADE"], errors='coerce')
                    df["FASE_IDEAL"] = np.clip(df["IDADE"] - 6, 1, 8)
                    self._created_features.append("FASE_IDEAL")
                except Exception:
                    logger.warning("Não foi possível processar coluna IDADE")
        
        # Calcular defasagem
        if "FASE" in df.columns and "FASE_IDEAL" in df.columns:
            # Garantir que FASE é numérica
            if not pd.api.types.is_numeric_dtype(df["FASE"]):
                df["FASE"] = pd.to_numeric(df["FASE"], errors='coerce')
            
            df["DEFASAGEM"] = df["FASE_IDEAL"] - df["FASE"]
            self._created_features.append("DEFASAGEM")
            
            # Defasagem absoluta
            df["DEFASAGEM_ABS"] = df["DEFASAGEM"].abs()
            self._created_features.append("DEFASAGEM_ABS")
            
            # Flag de defasagem
            df["TEM_DEFASAGEM"] = (df["DEFASAGEM"] > 0).astype(int)
            self._created_features.append("TEM_DEFASAGEM")
        
        # Usar coluna NIVEL_IDEAL se existir
        if "NIVEL_IDEAL" in df.columns and "FASE" in df.columns:
            # Garantir que NIVEL_IDEAL é numérica
            if not pd.api.types.is_numeric_dtype(df["NIVEL_IDEAL"]):
                df["NIVEL_IDEAL"] = pd.to_numeric(df["NIVEL_IDEAL"], errors='coerce')
            df["DEFASAGEM_NIVEL"] = df["NIVEL_IDEAL"] - df["FASE"]
            self._created_features.append("DEFASAGEM_NIVEL")
        
        logger.debug(f"Features de defasagem criadas")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features temporais.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features temporais
        """
        df = df.copy()
        
        # Anos na Passos Mágicos
        if "ANO_INGRESSO" in df.columns and "ANO_PEDE" in df.columns:
            df["ANOS_PM"] = df["ANO_PEDE"] - df["ANO_INGRESSO"]
            df["ANOS_PM"] = df["ANOS_PM"].clip(lower=0)
            self._created_features.append("ANOS_PM")
        elif "ANOS_PM_2024" in df.columns:
            df["ANOS_PM"] = df["ANOS_PM_2024"]
            self._created_features.append("ANOS_PM")
        
        # Veterano (mais de 2 anos)
        if "ANOS_PM" in df.columns:
            df["VETERANO"] = (df["ANOS_PM"] >= 2).astype(int)
            self._created_features.append("VETERANO")
        
        # Idade categorizada
        idade_col = "IDADE_CALC" if "IDADE_CALC" in df.columns else "IDADE"
        if idade_col in df.columns and pd.api.types.is_numeric_dtype(df[idade_col]):
            df["FAIXA_ETARIA"] = pd.cut(
                df[idade_col],
                bins=[0, 10, 13, 16, 20, 100],
                labels=["CRIANCA", "PRE_ADOLESCENTE", "ADOLESCENTE", "JOVEM", "ADULTO"]
            )
            self._created_features.append("FAIXA_ETARIA")
        
        logger.debug(f"Features temporais criadas")
        
        return df
    
    def create_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features derivadas dos indicadores.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features de indicadores
        """
        df = df.copy()
        
        # Lista de indicadores disponíveis
        available_indicators = [
            col for col in ["INDE", "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV"]
            if col in df.columns
        ]
        
        if not available_indicators:
            logger.warning("Nenhum indicador encontrado no DataFrame")
            return df
        
        # Garantir que indicadores são numéricos
        for col in available_indicators:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Média dos indicadores
        df["MEDIA_INDICADORES"] = df[available_indicators].mean(axis=1, skipna=True)
        self._created_features.append("MEDIA_INDICADORES")
        
        # Desvio padrão dos indicadores
        df["STD_INDICADORES"] = df[available_indicators].std(axis=1, skipna=True)
        self._created_features.append("STD_INDICADORES")
        
        # Coeficiente de variação
        df["CV_INDICADORES"] = safe_divide(
            df["STD_INDICADORES"],
            df["MEDIA_INDICADORES"],
            fill_value=0
        )
        self._created_features.append("CV_INDICADORES")
        
        # Mínimo e máximo dos indicadores
        df["MIN_INDICADOR"] = df[available_indicators].min(axis=1, skipna=True)
        df["MAX_INDICADOR"] = df[available_indicators].max(axis=1, skipna=True)
        self._created_features.extend(["MIN_INDICADOR", "MAX_INDICADOR"])
        
        # Range dos indicadores
        df["RANGE_INDICADORES"] = df["MAX_INDICADOR"] - df["MIN_INDICADOR"]
        self._created_features.append("RANGE_INDICADORES")
        
        # Contagem de indicadores baixos (< 5.5)
        df["INDICADORES_BAIXOS"] = (df[available_indicators] < 5.5).sum(axis=1)
        self._created_features.append("INDICADORES_BAIXOS")
        
        # Contagem de indicadores altos (>= 7.0)
        df["INDICADORES_ALTOS"] = (df[available_indicators] >= 7.0).sum(axis=1)
        self._created_features.append("INDICADORES_ALTOS")
        
        # Classificação PEDRA baseada no INDE
        if "INDE" in df.columns and "PEDRA" not in df.columns:
            df["PEDRA"] = df["INDE"].apply(categorize_inde)
            self._created_features.append("PEDRA")
        
        # INDE categorizado
        if "INDE" in df.columns:
            df["INDE_CATEGORIA"] = pd.cut(
                df["INDE"],
                bins=[0, 5.5, 6.5, 7.5, 10],
                labels=["BAIXO", "MEDIO", "BOM", "EXCELENTE"]
            )
            self._created_features.append("INDE_CATEGORIA")
        
        logger.debug(f"Features de indicadores criadas")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação entre variáveis.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features de interação
        """
        df = df.copy()
        
        # Ratio IDA/IEG (desempenho vs engajamento)
        if "IDA" in df.columns and "IEG" in df.columns:
            df["RATIO_IDA_IEG"] = safe_divide(df["IDA"], df["IEG"], fill_value=1)
            self._created_features.append("RATIO_IDA_IEG")
        
        # Ratio IPP/IPS (psicopedagógico vs psicossocial)
        if "IPP" in df.columns and "IPS" in df.columns:
            df["RATIO_IPP_IPS"] = safe_divide(df["IPP"], df["IPS"], fill_value=1)
            self._created_features.append("RATIO_IPP_IPS")
        
        # Engajamento x Desempenho
        if "IEG" in df.columns and "IDA" in df.columns:
            df["ENGAJ_X_DESEMP"] = df["IEG"] * df["IDA"]
            self._created_features.append("ENGAJ_X_DESEMP")
        
        # Score acadêmico (combinação de indicadores acadêmicos)
        academic_cols = [col for col in ["IDA", "IAN", "NOTA_PORT", "NOTA_MAT", "NOTA_ING"] if col in df.columns]
        if academic_cols:
            df["SCORE_ACADEMICO"] = df[academic_cols].mean(axis=1)
            self._created_features.append("SCORE_ACADEMICO")
        
        # Score comportamental
        behavior_cols = [col for col in ["IEG", "IPS", "IPP", "IAA"] if col in df.columns]
        if behavior_cols:
            df["SCORE_COMPORTAMENTAL"] = df[behavior_cols].mean(axis=1)
            self._created_features.append("SCORE_COMPORTAMENTAL")
        
        # Diferença entre scores
        if "SCORE_ACADEMICO" in df.columns and "SCORE_COMPORTAMENTAL" in df.columns:
            df["DIFF_ACAD_COMPORT"] = df["SCORE_ACADEMICO"] - df["SCORE_COMPORTAMENTAL"]
            self._created_features.append("DIFF_ACAD_COMPORT")
        
        # INDE x Anos na PM (maturidade)
        if "INDE" in df.columns and "ANOS_PM" in df.columns:
            df["INDE_X_ANOSPM"] = df["INDE"] * (df["ANOS_PM"] + 1)
            self._created_features.append("INDE_X_ANOSPM")
        
        # Flag de bolsista x ponto de virada
        if "BOLSISTA" in df.columns and "PONTO_VIRADA" in df.columns:
            df["BOLSISTA_E_VIRADA"] = (df["BOLSISTA"] & df["PONTO_VIRADA"]).astype(int)
            self._created_features.append("BOLSISTA_E_VIRADA")
        
        logger.debug(f"Features de interação criadas")
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria a variável target: RISCO_DEFASAGEM.
        
        Args:
            df: DataFrame com features
        
        Returns:
            DataFrame com target criado
        """
        df = df.copy()
        
        logger.info("Criando variável target RISCO_DEFASAGEM...")
        
        # Inicializar com risco baixo
        df["RISCO_DEFASAGEM"] = 0
        
        # Critérios para MÉDIO RISCO (1)
        conditions_medio = []
        
        # Defasagem == 1
        if "DEFASAGEM" in df.columns:
            conditions_medio.append(df["DEFASAGEM"] == 1)
        
        # INDE entre 5.5 e 6.5
        if "INDE" in df.columns:
            inde_medio = (df["INDE"] >= RISCO_THRESHOLDS["inde_alto_risco"]) & \
                        (df["INDE"] < RISCO_THRESHOLDS["inde_medio_risco"])
            conditions_medio.append(inde_medio)
        
        # Alguns indicadores baixos (2-3)
        if "INDICADORES_BAIXOS" in df.columns:
            conditions_medio.append((df["INDICADORES_BAIXOS"] >= 2) & (df["INDICADORES_BAIXOS"] <= 3))
        
        # Aplicar condições de médio risco
        if conditions_medio:
            medio_mask = pd.concat(conditions_medio, axis=1).any(axis=1)
            df.loc[medio_mask, "RISCO_DEFASAGEM"] = 1
        
        # Critérios para ALTO RISCO (2) - sobrescreve médio
        conditions_alto = []
        
        # Defasagem >= 2
        if "DEFASAGEM" in df.columns:
            conditions_alto.append(df["DEFASAGEM"] >= RISCO_THRESHOLDS["defasagem_alta"])
        
        # INDE < 5.5
        if "INDE" in df.columns:
            conditions_alto.append(df["INDE"] < RISCO_THRESHOLDS["inde_alto_risco"])
        
        # Muitos indicadores baixos (>= 4)
        if "INDICADORES_BAIXOS" in df.columns:
            conditions_alto.append(df["INDICADORES_BAIXOS"] >= 4)
        
        # Score acadêmico muito baixo
        if "SCORE_ACADEMICO" in df.columns:
            conditions_alto.append(df["SCORE_ACADEMICO"] < 5.0)
        
        # Aplicar condições de alto risco
        if conditions_alto:
            alto_mask = pd.concat(conditions_alto, axis=1).any(axis=1)
            df.loc[alto_mask, "RISCO_DEFASAGEM"] = 2
        
        # Criar label de risco
        df["NIVEL_RISCO"] = df["RISCO_DEFASAGEM"].map(RISCO_LABELS)
        
        self._created_features.extend(["RISCO_DEFASAGEM", "NIVEL_RISCO"])
        
        # Estatísticas do target
        risk_counts = df["RISCO_DEFASAGEM"].value_counts()
        logger.info(f"Distribuição do target:")
        for risk, count in risk_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {RISCO_LABELS[risk]}: {count} ({pct:.1f}%)")
        
        return df
    
    def get_created_features(self) -> List[str]:
        """
        Retorna lista de features criadas.
        
        Returns:
            Lista de nomes de features
        """
        return self._created_features.copy()


def calculate_risco_defasagem(row: pd.Series) -> int:
    """
    Calcula o risco de defasagem escolar para uma única linha.
    
    Critérios para classificação:
    - ALTO RISCO (2): DEFASAGEM >= 2 OU INDE < 5.5 OU múltiplos indicadores baixos
    - MÉDIO RISCO (1): DEFASAGEM == 1 OU (5.5 <= INDE < 6.5) OU tendência de queda
    - BAIXO RISCO (0): Sem defasagem E INDE >= 6.5 E indicadores estáveis/crescentes
    
    Args:
        row: Linha do DataFrame
    
    Returns:
        Código de risco (0, 1, ou 2)
    """
    # Extrair valores (com defaults)
    defasagem = row.get("DEFASAGEM", 0)
    inde = row.get("INDE", 7.0)
    indicadores_baixos = row.get("INDICADORES_BAIXOS", 0)
    
    # ALTO RISCO
    if (defasagem >= 2 or 
        inde < 5.5 or 
        indicadores_baixos >= 4):
        return 2
    
    # MÉDIO RISCO
    if (defasagem == 1 or 
        (5.5 <= inde < 6.5) or 
        (2 <= indicadores_baixos <= 3)):
        return 1
    
    # BAIXO RISCO
    return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função de conveniência para engenharia de features.
    
    Args:
        df: DataFrame original
    
    Returns:
        DataFrame com todas as features
    """
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)
