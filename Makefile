# Inception-of-Context  local runtime under /tmp/ioc
# Usage:
#   make setup   # create /tmp/ioc (venv, deps, embeddings, ollama model)
#   make p1      # run Part 1 (index + dashboard on :8000)
#   make p2      # stub  Part 2 (Ask & Retrieve)
#   make p3      # stub  Part 3 (Patch Loop)
#   make bonus   # stub  bonus features
#   make clean   # remove chroma db / caches under /tmp/ioc (keep venv)
#   make fclean  # full wipe of /tmp/ioc (venv + models cache + db)

IOC_DIR      := /tmp/ioc
VENV         := $(IOC_DIR)/venv
PIP_CACHE    := $(IOC_DIR)/pip-cache
HF_HOME      := $(IOC_DIR)/hf-cache
CHROMA_DIR   := $(IOC_DIR)/chroma_db
PYTHON       := /usr/bin/python3
PY_VER       := $(shell $(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
SITE_PACKAGES:= $(VENV)/lib/python$(PY_VER)/site-packages
REQ          := p1/requirements.txt
LLM_MODEL    := qwen2.5:3b
EMBED_MODEL  := all-MiniLM-L6-v2
PORT         := 8000
TARGET       := demo_app

# Campus image: python3 -m venv often lacks ensurepip; virtualenv is available.
VIRTUALENV   := $(shell command -v virtualenv 2>/dev/null)

export PIP_CACHE_DIR   := $(PIP_CACHE)
export HF_HOME
export TRANSFORMERS_CACHE := $(HF_HOME)
export TMPDIR          := /tmp
export PYTHONNOUSERSITE:= 1
export PYTHONPATH      := $(CURDIR):$(SITE_PACKAGES)

.PHONY: all setup p1 p2 p3 bonus stop clean fclean re help \
	ensure-dirs ensure-venv ensure-deps ensure-ready ensure-ollama \
	ensure-ollama-quick ensure-embed

all: setup

help:
	@echo "make setup  - prepare /tmp/ioc (venv, pip, embeddings, ollama $(LLM_MODEL))"
	@echo "make p1     - run Part 1 Overview dashboard on http://127.0.0.1:$(PORT)"
	@echo "make p2     - (stub) Part 2 Ask & Retrieve"
	@echo "make p3     - (stub) Part 3 Patch Loop"
	@echo "make bonus  - (stub) bonus features"
	@echo "make stop   - stop background ollama started by this Makefile (if any)"
	@echo "make clean  - remove chroma db / pip+hf caches (keep venv)"
	@echo "make fclean - full wipe of /tmp/ioc"
	@echo "make re     - fclean + setup"

ensure-dirs:
	@mkdir -p $(IOC_DIR) $(PIP_CACHE) $(HF_HOME) $(CHROMA_DIR)

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
	@if [ -f "$(IOC_DIR)/.deps-ok" ] && [ "$(IOC_DIR)/.deps-ok" -nt "$(REQ)" ] \
		&& $(PYTHON) -c "import chromadb,fastapi,uvicorn,watchdog,sentence_transformers" 2>/dev/null; then \
		echo "[*] Dependencies already ready (skip pip)"; \
	else \
		echo "[*] Installing CPU torch + Part 1 dependencies"; \
		$(VENV)/bin/pip install --upgrade pip setuptools wheel; \
		$(VENV)/bin/pip install --cache-dir $(PIP_CACHE) \
			torch --index-url https://download.pytorch.org/whl/cpu; \
		$(VENV)/bin/pip install --cache-dir $(PIP_CACHE) -r $(REQ) pillow; \
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

# Fast preflight for make p1 (no pip reinstall, no model re-pull if present)
ensure-ready: ensure-venv
	@$(PYTHON) -c "import chromadb,fastapi,uvicorn,watchdog,sentence_transformers" 2>/dev/null \
		|| { echo "[!] Runtime deps missing. Run: make setup"; exit 1; }

ensure-ollama: ensure-dirs
	@command -v ollama >/dev/null 2>&1 || { echo "ollama not found in PATH"; exit 1; }
	@if ! ollama list >/dev/null 2>&1; then \
		echo "[*] Starting ollama serve in background"; \
		mkdir -p $(IOC_DIR)/logs; \
		nohup ollama serve >$(IOC_DIR)/logs/ollama.log 2>&1 & echo $$! > $(IOC_DIR)/ollama.pid; \
		sleep 2; \
	fi
	@echo "[*] Ensuring Ollama model $(LLM_MODEL)"
	@ollama pull $(LLM_MODEL)

# Only start server if needed; pull model only when missing
ensure-ollama-quick: ensure-dirs
	@command -v ollama >/dev/null 2>&1 || { echo "ollama not found in PATH"; exit 1; }
	@if ! ollama list >/dev/null 2>&1; then \
		echo "[*] Starting ollama serve in background"; \
		mkdir -p $(IOC_DIR)/logs; \
		nohup ollama serve >$(IOC_DIR)/logs/ollama.log 2>&1 & echo $$! > $(IOC_DIR)/ollama.pid; \
		sleep 2; \
	fi
	@if ! ollama show $(LLM_MODEL) >/dev/null 2>&1; then \
		echo "[*] Pulling missing model $(LLM_MODEL)"; \
		ollama pull $(LLM_MODEL); \
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

p2:
	@echo "[!] Part 2 (Ask & Retrieve) is not implemented yet."
	@echo "    Planned: POST /context, POST /ask, Ollama RAG UI."
	@exit 1

p3:
	@echo "[!] Part 3 (Patch Loop) is not implemented yet."
	@echo "    Planned: structured patch JSON, validation, atomic rollback."
	@exit 1

bonus:
	@echo "[!] Bonus features are not implemented yet."
	@exit 1

stop:
	@if [ -f $(IOC_DIR)/ollama.pid ]; then \
		kill $$(cat $(IOC_DIR)/ollama.pid) 2>/dev/null || true; \
		rm -f $(IOC_DIR)/ollama.pid; \
		echo "[*] Stopped Makefile-started ollama"; \
	else \
		echo "[*] No Makefile ollama pid file"; \
	fi

clean: stop
	@rm -rf $(CHROMA_DIR) $(PIP_CACHE) $(HF_HOME) $(IOC_DIR)/logs $(IOC_DIR)/.deps-ok
	@echo "[*] Cleaned caches and chroma db under $(IOC_DIR) (venv kept)"

fclean: stop
	@rm -rf $(IOC_DIR)
	@echo "[*] Removed $(IOC_DIR)"

re: fclean setup
