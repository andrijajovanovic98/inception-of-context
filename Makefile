# Inception-of-Context  local runtime under /tmp/ioc
# Usage:
#   make setup   # create /tmp/ioc (venv, deps, embeddings, ollama model)
#   make p1      # run Part 1 (index + dashboard on :8000)
#   make p2      # stub  Part 2 (Ask & Retrieve)
#   make p3      # stub  Part 3 (Patch Loop)
#   make bonus   # stub  bonus features
#   make flake   # run flake8 on Python packages
#   make mypy    # run mypy on Python packages
#   make lint    # run flake8 + mypy
#   make clean   # remove chroma db / caches under /tmp/ioc (keep venv)
#   make fclean  # full wipe of /tmp/ioc (venv + models cache + db)

IOC_DIR      := /tmp/ioc
VENV         := $(IOC_DIR)/venv
PIP_CACHE    := $(IOC_DIR)/pip-cache
HF_HOME      := $(IOC_DIR)/hf-cache
CHROMA_DIR   := $(IOC_DIR)/chroma_db
OLLAMA_DIR   := $(IOC_DIR)/ollama
# Private port so we do not reuse the campus server that writes to /opt/ollama.
# Listen on all interfaces so Docker (host.docker.internal) can reach Ollama.
OLLAMA_BIND  := 0.0.0.0:11435
# Clients (local make p1/p2/p3/bonus + ollama CLI) still use loopback.
OLLAMA_HOST  := 127.0.0.1:11435
PYTHON       := /usr/bin/python3
PY_VER       := $(shell $(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITE_PACKAGES:= $(VENV)/lib/python$(PY_VER)/site-packages
REQ          := p1/requirements.txt
REQ_P2       := p2/requirements.txt
REQ_P3       := p3/requirements.txt
LLM_MODEL    := qwen2.5:3b
EMBED_MODEL  := all-MiniLM-L6-v2
PORT         := 8000
TARGET       := demo_app
LINT_DIRS    := p1 p2 p3 bonus demo_app

# Campus image: python3 -m venv often lacks ensurepip; virtualenv is available.
VIRTUALENV   := $(shell command -v virtualenv 2>/dev/null)

export PIP_CACHE_DIR   := $(PIP_CACHE)
export HF_HOME
export TRANSFORMERS_CACHE := $(HF_HOME)
export TMPDIR          := /tmp
export PYTHONNOUSERSITE:= 1
export PYTHONPATH      := $(CURDIR):$(SITE_PACKAGES)
# Writable models dir (overrides campus OLLAMA_MODELS=/opt/ollama)
export OLLAMA_MODELS   := $(OLLAMA_DIR)
export OLLAMA_HOST

.PHONY: all up down setup p1 p2 p3 p3-cli bonus stop clean fclean re help \
	ensure-dirs ensure-venv ensure-deps ensure-ready ensure-ollama \
	ensure-ollama-quick ensure-embed ensure-lint-tools flake mypy lint

all: setup

up:
	@docker compose up --build -d 2>/dev/null || docker-compose up --build -d

down:
	@docker compose down 2>/dev/null || docker-compose down

help:
	@echo "make up      - build and launch containerized IoC via Docker Compose"
	@echo "make down    - stop and tear down Docker containers"
	@echo "make setup   - prepare /tmp/ioc (venv, pip, embeddings, ollama $(LLM_MODEL))"
	@echo "make p1      - run Part 1 Overview dashboard on http://127.0.0.1:$(PORT)"
	@echo "make p2      - run Part 2 Architect API & RAG dashboard on http://127.0.0.1:$(PORT)"
	@echo "make p3      - run Part 3 Patch Loop & Dashboard on http://127.0.0.1:$(PORT)"
	@echo "make p3-cli  - run headless patch loop: make p3-cli INTENT=\"your intent\""
	@echo "make bonus   - run Chapter VII Bonus Suite & Dashboard on http://127.0.0.1:$(PORT)"
	@echo "make flake   - run flake8 on $(LINT_DIRS)"
	@echo "make mypy    - run mypy on $(LINT_DIRS)"
	@echo "make lint    - run flake8 + mypy on $(LINT_DIRS)"
	@echo "make stop    - stop background ollama started by this Makefile (if any)"
	@echo "make clean   - remove chroma db / pip+hf caches (keep venv)"
	@echo "make fclean  - full wipe of /tmp/ioc"
	@echo "make re      - fclean + setup"

ensure-dirs:
	@mkdir -p $(IOC_DIR) $(PIP_CACHE) $(HF_HOME) $(CHROMA_DIR) $(OLLAMA_DIR)

ensure-venv: ensure-dirs
	@if [ ! -x "$(VENV)/bin/pip" ]; then \
		if [ -z "$(VIRTUALENV)" ]; then \
			echo "virtualenv not found. Install it or use: python3 -m venv $(VENV)"; \
			exit 1; \
		fi; \
		echo "[*] Creating virtualenv at $(VENV)"; \
		PYTHONPATH="$$HOME/.local/lib/python$(PY_VER)/site-packages/setuptools/_vendor:$${PYTHONPATH:-}" \
			$(VIRTUALENV) $(VENV); \
	else \
		echo "[*] Virtualenv already present: $(VENV)"; \
	fi

ensure-deps: ensure-venv
	@if [ -f "$(IOC_DIR)/.deps-ok" ] && [ "$(IOC_DIR)/.deps-ok" -nt "$(REQ)" ] && [ "$(IOC_DIR)/.deps-ok" -nt "$(REQ_P2)" ] && [ "$(IOC_DIR)/.deps-ok" -nt "$(REQ_P3)" ] \
		&& $(PYTHON) -c "import chromadb,fastapi,uvicorn,watchdog,sentence_transformers,rank_bm25,httpx,yaml" 2>/dev/null; then \
		echo "[*] Dependencies already ready (skip pip)"; \
	else \
		echo "[*] Installing CPU torch + Part 1, 2 & 3 dependencies"; \
		$(VENV)/bin/pip install --upgrade pip setuptools wheel; \
		$(VENV)/bin/pip install --cache-dir $(PIP_CACHE) \
			torch --index-url https://download.pytorch.org/whl/cpu; \
		$(VENV)/bin/pip install --cache-dir $(PIP_CACHE) -r $(REQ) -r $(REQ_P2) -r $(REQ_P3) pillow; \
		touch "$(IOC_DIR)/.deps-ok"; \
	fi
	@printf '%s\n' \
		'#!/bin/sh' \
		'set -eu' \
		'ROOT="$(CURDIR)"' \
		'export PYTHONNOUSERSITE=1' \
		'export PYTHONPATH="$(CURDIR):$(SITE_PACKAGES)"' \
		'export HF_HOME="$(HF_HOME)"' \
		'export TRANSFORMERS_CACHE="$(HF_HOME)"' \
		'export TMPDIR=/tmp' \
		'cd "$$ROOT"' \
		'exec /usr/bin/python3 "$$ROOT/p1/index.py" "$$@"' \
		> $(IOC_DIR)/run.sh
	@chmod +x $(IOC_DIR)/run.sh

# Fast preflight for make p1, p2, p3 (no pip install, no empty venv creation)
ensure-ready:
	@if [ ! -x "$(VENV)/bin/pip" ] \
		|| ! $(PYTHON) -c "import chromadb,fastapi,uvicorn,watchdog,sentence_transformers,rank_bm25,httpx,yaml" 2>/dev/null; then \
		echo "[!] IoC runtime not ready under $(IOC_DIR) (missing after fclean, or never set up)."; \
		echo "    1) make setup"; \
		echo "    2) make p1, make p2, or make p3"; \
		exit 1; \
	fi
	@echo "[*] Runtime ready: $(VENV)"

ensure-ollama: ensure-dirs
	@command -v ollama >/dev/null 2>&1 || { echo "ollama not found in PATH"; exit 1; }
	@if [ -f $(IOC_DIR)/ollama.pid ] && kill -0 $$(cat $(IOC_DIR)/ollama.pid) 2>/dev/null; then \
		echo "[*] Ollama already running (pid $$(cat $(IOC_DIR)/ollama.pid), bind $(OLLAMA_BIND))"; \
	else \
		echo "[*] Starting ollama serve (models=$(OLLAMA_DIR), bind=$(OLLAMA_BIND))"; \
		mkdir -p $(IOC_DIR)/logs $(OLLAMA_DIR); \
		nohup env HOME="$(IOC_DIR)" OLLAMA_MODELS="$(OLLAMA_DIR)" OLLAMA_HOST="$(OLLAMA_BIND)" \
			ollama serve >$(IOC_DIR)/logs/ollama.log 2>&1 & echo $$! > $(IOC_DIR)/ollama.pid; \
		sleep 2; \
	fi
	@echo "[*] Ensuring Ollama model $(LLM_MODEL)"
	@HOME="$(IOC_DIR)" OLLAMA_MODELS="$(OLLAMA_DIR)" OLLAMA_HOST="$(OLLAMA_HOST)" ollama pull $(LLM_MODEL)

# Only start server if needed; pull model only when missing
ensure-ollama-quick: ensure-dirs
	@command -v ollama >/dev/null 2>&1 || { echo "ollama not found in PATH"; exit 1; }
	@if [ -f $(IOC_DIR)/ollama.pid ] && kill -0 $$(cat $(IOC_DIR)/ollama.pid) 2>/dev/null; then \
		true; \
	else \
		echo "[*] Starting ollama serve (models=$(OLLAMA_DIR), bind=$(OLLAMA_BIND))"; \
		mkdir -p $(IOC_DIR)/logs $(OLLAMA_DIR); \
		nohup env HOME="$(IOC_DIR)" OLLAMA_MODELS="$(OLLAMA_DIR)" OLLAMA_HOST="$(OLLAMA_BIND)" \
			ollama serve >$(IOC_DIR)/logs/ollama.log 2>&1 & echo $$! > $(IOC_DIR)/ollama.pid; \
		sleep 2; \
	fi
	@if ! HOME="$(IOC_DIR)" OLLAMA_MODELS="$(OLLAMA_DIR)" OLLAMA_HOST="$(OLLAMA_HOST)" \
		ollama show $(LLM_MODEL) >/dev/null 2>&1; then \
		echo "[*] Pulling missing model $(LLM_MODEL)"; \
		HOME="$(IOC_DIR)" OLLAMA_MODELS="$(OLLAMA_DIR)" OLLAMA_HOST="$(OLLAMA_HOST)" ollama pull $(LLM_MODEL); \
	fi

ensure-embed: ensure-deps
	@echo "[*] Prefetching embedding model $(EMBED_MODEL) into $(HF_HOME)"
	@$(PYTHON) -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$(EMBED_MODEL)'); print('[+] embedding model ready')"

setup: ensure-deps ensure-ollama ensure-embed
	@echo
	@echo "[+] Setup complete under $(IOC_DIR)"
	@echo "    Next: make p1   →  http://127.0.0.1:$(PORT)"

# ---------------------------------------------------------------------------
# Part runners
# ---------------------------------------------------------------------------

p1: ensure-ready ensure-ollama-quick
	@echo "[*] Part 1 - indexing $(TARGET) and starting dashboard on :$(PORT)"
	@$(PYTHON) p1/index.py $(TARGET) \
		--db-dir $(CHROMA_DIR) \
		--watch --dashboard \
		--host 127.0.0.1 --port $(PORT) \
		--llm-model $(LLM_MODEL)

p2: ensure-ready ensure-ollama-quick
	@echo "[*] Part 2 - Architect API & RAG (Ask & Retrieve) on :$(PORT)"
	@$(PYTHON) p2/index.py $(TARGET) \
		--db-dir $(CHROMA_DIR) \
		--watch --dashboard \
		--host 127.0.0.1 --port $(PORT) \
		--llm-model $(LLM_MODEL) \
		--ollama-host $(OLLAMA_HOST)

p3: ensure-ready ensure-ollama-quick
	@echo "[*] Part 3 - Autonomous Patch Loop & Dashboard on :$(PORT)"
	@$(PYTHON) p3/index.py $(TARGET) \
		--db-dir $(CHROMA_DIR) \
		--watch --dashboard \
		--host 127.0.0.1 --port $(PORT) \
		--llm-model $(LLM_MODEL) \
		--ollama-host $(OLLAMA_HOST)

p3-cli: ensure-ready ensure-ollama-quick
	@if [ -z "$(INTENT)" ]; then \
		echo "Usage: make p3-cli INTENT=\"your coding intent\""; \
		exit 1; \
	fi
	@$(PYTHON) p3/index.py $(TARGET) \
		--db-dir $(CHROMA_DIR) \
		--intent "$(INTENT)" \
		--llm-model $(LLM_MODEL) \
		--ollama-host $(OLLAMA_HOST)

bonus: ensure-ready ensure-ollama-quick
	@echo "[*] Running Chapter VII Bonus Suite on :$(PORT)"
	@$(PYTHON) bonus/index.py $(TARGET) \
		--db-dir $(CHROMA_DIR) \
		--watch --dashboard \
		--host 127.0.0.1 --port $(PORT) \
		--llm-model $(LLM_MODEL) \
		--ollama-host $(OLLAMA_HOST)

stop:
	@if [ -f $(IOC_DIR)/ollama.pid ]; then \
		kill $$(cat $(IOC_DIR)/ollama.pid) 2>/dev/null || true; \
		rm -f $(IOC_DIR)/ollama.pid; \
		echo "[*] Stopped Makefile-started ollama"; \
	else \
		echo "[*] No Makefile ollama pid file"; \
	fi

# ---------------------------------------------------------------------------
# Lint (flake8 + mypy)
# ---------------------------------------------------------------------------

ensure-lint-tools: ensure-venv
	@if $(VENV)/bin/python -c "import flake8, mypy" 2>/dev/null; then \
		echo "[*] Lint tools already installed in $(VENV)"; \
	else \
		echo "[*] Installing flake8 + mypy into $(VENV)"; \
		$(VENV)/bin/pip install --cache-dir $(PIP_CACHE) flake8 mypy types-PyYAML; \
	fi

flake: ensure-lint-tools
	@echo "[*] flake8 → $(LINT_DIRS)"
	@$(VENV)/bin/python -m flake8 $(LINT_DIRS)

mypy: ensure-lint-tools
	@echo "[*] mypy → $(LINT_DIRS)"
	@PYTHONPATH="$(CURDIR):$(SITE_PACKAGES)" $(VENV)/bin/python -m mypy \
		--config-file mypy.ini $(LINT_DIRS)

lint: flake mypy
	@echo "[+] lint OK (flake8 + mypy)"

clean: stop
	@rm -rf $(CHROMA_DIR) $(PIP_CACHE) $(HF_HOME) $(IOC_DIR)/logs $(IOC_DIR)/.deps-ok
	@echo "[*] Cleaned caches and chroma db under $(IOC_DIR) (venv kept)"

fclean: stop
	@rm -rf $(IOC_DIR)
	@echo "[*] Removed $(IOC_DIR)"

re: fclean setup
