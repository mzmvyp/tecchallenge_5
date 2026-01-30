.PHONY: install install-dev test test-cov train run run-dev docker-build docker-run docker-stop lint format clean docs help

# Variáveis
PYTHON = python
PIP = pip
PYTEST = pytest
UVICORN = uvicorn
DOCKER = docker
DOCKER_COMPOSE = docker-compose

# Cores para output
GREEN = \033[0;32m
NC = \033[0m

help:
	@echo "Comandos disponíveis:"
	@echo "  make install      - Instala dependências de produção"
	@echo "  make install-dev  - Instala dependências de desenvolvimento"
	@echo "  make test         - Executa testes"
	@echo "  make test-cov     - Executa testes com cobertura"
	@echo "  make train        - Treina o modelo"
	@echo "  make run          - Executa a API em modo produção"
	@echo "  make run-dev      - Executa a API em modo desenvolvimento"
	@echo "  make docker-build - Constrói imagem Docker"
	@echo "  make docker-run   - Executa container Docker"
	@echo "  make docker-stop  - Para container Docker"
	@echo "  make lint         - Executa linting"
	@echo "  make format       - Formata código"
	@echo "  make clean        - Limpa arquivos temporários"
	@echo "  make docs         - Gera documentação"

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ -v --cov=src --cov=api --cov-report=html --cov-report=term --cov-fail-under=80

train:
	$(PYTHON) -m src.models.train

run:
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000

run-dev:
	$(UVICORN) api.main:app --reload --host 127.0.0.1 --port 8000

docker-build:
	$(DOCKER) build -t passos-magicos-ml .

docker-run:
	$(DOCKER_COMPOSE) up -d

docker-stop:
	$(DOCKER_COMPOSE) down

lint:
	flake8 src/ api/ tests/
	mypy src/ api/

format:
	black src/ api/ tests/
	isort src/ api/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Comandos de desenvolvimento
eda:
	jupyter notebook notebooks/01_eda.ipynb

preprocess:
	$(PYTHON) -c "from src.preprocessing import DataLoader, DataCleaner, DataTransformer; print('Preprocessing modules OK')"

# Verificação de saúde
check:
	@echo "Verificando dependências..."
	$(PYTHON) -c "import pandas; import sklearn; import fastapi; print('Todas as dependências OK')"
	@echo "Verificando estrutura do projeto..."
	@test -d src && echo "  ✓ src/" || echo "  ✗ src/ não encontrado"
	@test -d api && echo "  ✓ api/" || echo "  ✗ api/ não encontrado"
	@test -d tests && echo "  ✓ tests/" || echo "  ✗ tests/ não encontrado"
	@test -d data/raw && echo "  ✓ data/raw/" || echo "  ✗ data/raw/ não encontrado"
