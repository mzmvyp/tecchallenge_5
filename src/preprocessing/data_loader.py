"""
Módulo para carregamento de dados do projeto Passos Mágicos.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.config import RAW_DATA_DIR, SHEETS, settings
from src.utils.logger import get_logger
from src.utils.helpers import clean_column_names, get_memory_usage

logger = get_logger(__name__)


class DataLoader:
    """Classe para carregamento e unificação de dados."""
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        Inicializa o DataLoader.
        
        Args:
            data_path: Caminho para o arquivo de dados (Excel)
        """
        if data_path is None:
            data_path = Path(settings.data_file)
        
        self.data_path = Path(data_path)
        self._validate_path()
        
        logger.info(f"DataLoader inicializado com: {self.data_path}")
    
    def _validate_path(self) -> None:
        """Valida se o arquivo de dados existe."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Arquivo de dados não encontrado: {self.data_path}"
            )
    
    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """
        Carrega uma sheet específica do Excel.
        
        Args:
            sheet_name: Nome da sheet a ser carregada
        
        Returns:
            DataFrame com os dados da sheet
        """
        logger.info(f"Carregando sheet: {sheet_name}")
        
        df = pd.read_excel(
            self.data_path,
            sheet_name=sheet_name,
            engine="openpyxl"
        )
        
        # Limpar nomes das colunas
        df = clean_column_names(df)
        
        # Adicionar coluna de ano/período
        year = self._extract_year_from_sheet_name(sheet_name)
        df["ANO_PEDE"] = year
        
        logger.info(
            f"Sheet {sheet_name} carregada: {len(df)} registros, "
            f"{len(df.columns)} colunas, {get_memory_usage(df)}"
        )
        
        return df
    
    def _extract_year_from_sheet_name(self, sheet_name: str) -> int:
        """
        Extrai o ano do nome da sheet.
        
        Args:
            sheet_name: Nome da sheet (ex: "PEDE_2022")
        
        Returns:
            Ano extraído
        """
        # Procurar por padrão de ano (4 dígitos)
        import re
        match = re.search(r"20\d{2}", sheet_name)
        if match:
            return int(match.group())
        
        # Padrão: último ano conhecido
        logger.warning(f"Não foi possível extrair ano de '{sheet_name}', usando 2024")
        return 2024
    
    def load_all_sheets(
        self,
        sheet_names: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Carrega todas as sheets do arquivo Excel.
        
        Args:
            sheet_names: Lista de sheets a carregar (None = todas disponíveis)
        
        Returns:
            Dicionário {nome_sheet: DataFrame}
        """
        if sheet_names is None:
            # Descobrir sheets disponíveis
            xlsx = pd.ExcelFile(self.data_path, engine="openpyxl")
            sheet_names = xlsx.sheet_names
            logger.info(f"Sheets disponíveis: {sheet_names}")
        
        data = {}
        for sheet in sheet_names:
            try:
                data[sheet] = self.load_sheet(sheet)
            except Exception as e:
                logger.error(f"Erro ao carregar sheet '{sheet}': {e}")
                continue
        
        return data
    
    def load_and_unify(
        self,
        sheet_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Carrega e unifica todas as sheets em um único DataFrame.
        
        Args:
            sheet_names: Lista de sheets a carregar
        
        Returns:
            DataFrame unificado com todos os dados
        """
        logger.info("Iniciando carregamento e unificação dos dados...")
        
        sheets_data = self.load_all_sheets(sheet_names)
        
        if not sheets_data:
            raise ValueError("Nenhuma sheet foi carregada com sucesso")
        
        # Encontrar colunas comuns
        all_columns = [set(df.columns) for df in sheets_data.values()]
        common_columns = set.intersection(*all_columns)
        
        logger.info(f"Colunas comuns encontradas: {len(common_columns)}")
        
        # Unificar DataFrames
        unified_dfs = []
        for sheet_name, df in sheets_data.items():
            # Adicionar colunas faltantes como NaN
            for col in common_columns - set(df.columns):
                df[col] = pd.NA
            unified_dfs.append(df)
        
        df_unified = pd.concat(unified_dfs, ignore_index=True)
        
        logger.info(
            f"Dados unificados: {len(df_unified)} registros, "
            f"{len(df_unified.columns)} colunas, {get_memory_usage(df_unified)}"
        )
        
        return df_unified
    
    def get_column_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna informações sobre as colunas do DataFrame.
        
        Args:
            df: DataFrame para analisar
        
        Returns:
            DataFrame com informações das colunas
        """
        info = []
        
        for col in df.columns:
            info.append({
                "coluna": col,
                "tipo": str(df[col].dtype),
                "nulos": df[col].isnull().sum(),
                "nulos_pct": f"{df[col].isnull().mean() * 100:.2f}%",
                "unicos": df[col].nunique(),
                "exemplo": df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            })
        
        return pd.DataFrame(info)
    
    def save_processed(
        self,
        df: pd.DataFrame,
        filename: str = "unified_data.parquet"
    ) -> Path:
        """
        Salva DataFrame processado.
        
        Args:
            df: DataFrame a ser salvo
            filename: Nome do arquivo
        
        Returns:
            Caminho do arquivo salvo
        """
        from src.config import PROCESSED_DATA_DIR
        from src.utils.helpers import ensure_dir
        
        ensure_dir(PROCESSED_DATA_DIR)
        output_path = PROCESSED_DATA_DIR / filename
        
        # Salvar em formato parquet para melhor performance
        if filename.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        elif filename.endswith(".csv"):
            df.to_csv(output_path, index=False, encoding="utf-8")
        else:
            df.to_parquet(output_path.with_suffix(".parquet"), index=False)
        
        logger.info(f"Dados salvos em: {output_path}")
        
        return output_path


def load_data(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Função de conveniência para carregar e unificar dados.
    
    Args:
        data_path: Caminho para o arquivo de dados
    
    Returns:
        DataFrame unificado
    """
    loader = DataLoader(data_path)
    return loader.load_and_unify()
