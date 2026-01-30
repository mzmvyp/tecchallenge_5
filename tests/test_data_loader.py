"""
Testes para o módulo data_loader.
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.preprocessing.data_loader import DataLoader, load_data


class TestDataLoader:
    """Testes para a classe DataLoader."""
    
    def test_init_with_default_path(self):
        """Testa inicialização com caminho padrão."""
        loader = DataLoader()
        assert loader.data_path is not None
        assert "BASE DE DADOS PEDE 2024" in str(loader.data_path)
    
    def test_load_sheet_returns_dataframe(self):
        """Testa que load_sheet retorna DataFrame."""
        loader = DataLoader()
        
        if loader.data_path.exists():
            # Usar a primeira sheet disponível
            df = loader.load_sheet("PEDE2024")
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_load_and_unify_combines_sheets(self):
        """Testa que load_and_unify combina sheets corretamente."""
        loader = DataLoader()
        
        if loader.data_path.exists():
            df = loader.load_and_unify()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            # Deve ter coluna ANO_PEDE após unificação
            assert "ANO_PEDE" in df.columns
    
    def test_load_all_sheets_returns_dict(self):
        """Testa que load_all_sheets retorna dicionário."""
        loader = DataLoader()
        
        if loader.data_path.exists():
            sheets = loader.load_all_sheets()
            assert isinstance(sheets, dict)
            assert len(sheets) > 0
    
    def test_get_column_info_returns_dataframe(self):
        """Testa que get_column_info retorna DataFrame com informações."""
        loader = DataLoader()
        
        if loader.data_path.exists():
            df = loader.load_sheet("PEDE2024")
            info = loader.get_column_info(df)
            
            assert isinstance(info, pd.DataFrame)
            assert len(info) > 0


class TestDataLoaderEdgeCases:
    """Testes de casos extremos para DataLoader."""
    
    def test_load_sheet_nonexistent(self):
        """Testa carregar sheet que não existe."""
        loader = DataLoader()
        
        if loader.data_path.exists():
            with pytest.raises(Exception):  # Pode ser ValueError ou KeyError
                loader.load_sheet("SHEET_QUE_NAO_EXISTE_12345")


class TestDataLoaderWithMock:
    """Testes com mock para DataLoader."""
    
    def test_load_sheet_with_mock(self, tmp_path):
        """Testa load_sheet com arquivo mock."""
        # Criar arquivo Excel de teste
        test_file = tmp_path / "test.xlsx"
        
        df_test = pd.DataFrame({
            "NOME": ["Aluno A", "Aluno B"],
            "INDE": [7.5, 6.0],
            "FASE": [5, 3]
        })
        
        df_test.to_excel(test_file, sheet_name="Sheet1", index=False)
        
        loader = DataLoader(data_path=test_file)
        df = loader.load_sheet("Sheet1")
        
        assert len(df) == 2
        assert "NOME" in df.columns
        assert "INDE" in df.columns


class TestLoadDataFunction:
    """Testes para a função load_data."""
    
    def test_load_data_default(self):
        """Testa função de conveniência load_data."""
        df = load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
