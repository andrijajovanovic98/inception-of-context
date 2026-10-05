# Inception-of-Context  local runtime under /tmp/ioc
# Usage:
#   make setup        # create /tmp/ioc (venv, deps, embeddings, ollama model)
#   make p1 / p2 / p3 / bonus
#   make flake / mypy / lint / test / test-p1 / test-p2 / test-p3 / test-bonus / gates
#   make up / down / docker-restart  # Docker Compose
#   make docker-clean / docker-fclean
#   make stop / clean / fclean / re

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
REQ_BONUS    := bonus/requirements.txt
LLM_MODEL    := qwen2.5:3b
EMBED_MODEL  := all-MiniLM-L6-v2
EMBED_CACHE  := $(HF_HOME)/hub/models--sentence-transformers--$(EMBED_MODEL)
EMBED_FETCH  := $(PYTHON) -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$(EMBED_MODEL)'); print('[+] embedding model ready')"
PORT         := 8000
TARGET       := demo_app
LINT_DIRS    := p1 p2 p3 bonus demo_app
TEST_DIRS    := p1/tests p2/tests p3/tests bonus/tests
TEST_ARGS    ?= -v
DOCKER_IMAGE := inception-of-context-ioc
DEMO_SERVICE := ioc-demo-service
DEMO_IMAGE   ?= python:3.10-slim
DEMO_INTERVAL ?= 5
WATCH_CONTAINER ?=

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

.PHONY: all up down docker-restart docker-clean docker-fclean setup p1 p2 p3 p3-cli \
	bonus stop clean fclean re help ensure-dirs ensure-venv ensure-deps ensure-ready \
	ensure-ollama ensure-ollama-quick ensure-ollama-soft ensure-embed ensure-test-runtime ensure-lint-tools \
	flake mypy lint test test-p1 test-p2 test-p3 test-bonus gates demo-service demo-service-stop

all: setup

# Compose v2 plugin when present, standalone docker-compose otherwise. Chosen
# once: "v2 2>/dev/null || v1" hid a real build error and then ran the whole
# build a second time through v1.
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

# Container user (docker-compose.yml `user:`). Rootful Docker: the invoking
# user, so files the patch loop writes into the bind-mounted demo_app/ and
# $(CHROMA_DIR) stay theirs instead of becoming root-owned. Rootless Docker:
# 0:0, because container root already maps to the invoking user there and any
# other uid is unmapped (the container cannot even start).
IOC_USER = $(shell if docker info --format '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless; \
	then echo 0:0; else echo "$$(id -u):$$(id -g)"; fi)

up: ensure-ollama-soft
	@# The bind-mount source must exist BEFORE compose runs, or Docker creates
	@# it as root and a non-root container cannot write the vector store.
	@mkdir -p $(CHROMA_DIR)
	@IOC_USER=$(IOC_USER) $(COMPOSE) up --build -d
	@echo "[+] IoC is up: http://127.0.0.1:$(PORT)  (logs: $(COMPOSE) logs -f ioc)"

down:
	@$(COMPOSE) down

# Rebuild image (bonus/p* are not volume-mounted) and recreate the container.
docker-restart: ensure-ollama-soft
	@mkdir -p $(CHROMA_DIR)
	@IOC_USER=$(IOC_USER) $(COMPOSE) up --build -d --force-recreate
	@echo "[+] docker-restart done (rebuild + recreate ioc-app)"

# The container talks to the host's IoC Ollama on $(OLLAMA_HOST). Start it
# when it is installed but not running (e.g. after a reboot). Never fails and
# never pulls: a Docker user may run their own Ollama on that port.
ensure-ollama-soft: ensure-dirs
	@if ! command -v ollama >/dev/null 2>&1; then \
		echo "[!] ollama not in PATH: the container expects one on $(OLLAMA_HOST) (see make setup)"; \
	elif [ -f $(IOC_DIR)/ollama.pid ] && kill -0 $$(cat $(IOC_DIR)/ollama.pid) 2>/dev/null; then \
		true; \
	else \
		echo "[*] Starting ollama serve for the container (models=$(OLLAMA_DIR), bind=$(OLLAMA_BIND))"; \
		mkdir -p $(IOC_DIR)/logs $(OLLAMA_DIR); \
		nohup env HOME="$(IOC_DIR)" OLLAMA_MODELS="$(OLLAMA_DIR)" OLLAMA_HOST="$(OLLAMA_BIND)" \
			ollama serve >$(IOC_DIR)/logs/ollama.log 2>&1 & echo $$! > $(IOC_DIR)/ollama.pid; \
		sleep 2; \
	fi
	@HOME="$(IOC_DIR)" OLLAMA_MODELS="$(OLLAMA_DIR)" OLLAMA_HOST="$(OLLAMA_HOST)" \
		ollama show $(LLM_MODEL) >/dev/null 2>&1 \
		|| echo "[!] $(LLM_MODEL) is not available on $(OLLAMA_HOST) yet - run: make setup"

# Soft Docker cleanup (IoC only): stop/remove container + project network, keep image.
# No error if Docker is missing or IoC was never built.
docker-clean:
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "[*] Docker not available - skip docker-clean"; \
	else \
		echo "[*] Docker clean (container/network; keep image $(DOCKER_IMAGE))"; \
		$(COMPOSE) down --remove-orphans >/dev/null 2>&1 || true; \
		if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx '$(DEMO_SERVICE)'; then \
			docker rm -f $(DEMO_SERVICE) >/dev/null 2>&1 || true; \
			echo "[*] Removed container $(DEMO_SERVICE)"; \
		fi; \
		if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx 'ioc-app'; then \
			docker rm -f ioc-app >/dev/null 2>&1 || true; \
			echo "[*] Removed container ioc-app"; \
		fi; \
		echo "[+] docker-clean done"; \
	fi

# Full Docker cleanup (IoC only): container, networks, volumes, image.
# Prunes only if they exist; never fails when Docker/IoC artifacts are absent.
docker-fclean:
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "[*] Docker not available - skip docker-fclean"; \
	else \
		echo "[*] Docker fclean (container/network/volume/image for IoC only)"; \
		$(COMPOSE) down --rmi local --volumes --remove-orphans >/dev/null 2>&1 || true; \
		if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx 'ioc-app'; then \
			docker rm -f ioc-app >/dev/null 2>&1 || true; \
			echo "[*] Removed container ioc-app"; \
		fi; \
		if docker image inspect $(DOCKER_IMAGE):latest >/dev/null 2>&1; then \
			docker rmi -f $(DOCKER_IMAGE):latest >/dev/null 2>&1 || true; \
			echo "[*] Removed image $(DOCKER_IMAGE):latest"; \
		fi; \
		ids=$$(docker images -q '$(DOCKER_IMAGE)' 2>/dev/null || true); \
		if [ -n "$$ids" ]; then \
			docker rmi -f $$ids >/dev/null 2>&1 || true; \
			echo "[*] Removed remaining $(DOCKER_IMAGE) image tags"; \
		fi; \
		if docker network ls --format '{{.Name}}' 2>/dev/null | grep -qx 'inception-of-context_default'; then \
			docker network rm inception-of-context_default >/dev/null 2>&1 || true; \
			echo "[*] Removed network inception-of-context_default"; \
		fi; \
		echo "[+] docker-fclean done (other Docker images untouched)"; \
	fi

help:
	@echo "make up               - build and launch containerized IoC via Docker Compose"
	@echo "make down             - stop and tear down Docker containers"
	@echo "make docker-restart   - rebuild image and recreate ioc-app (code changes)"
	@echo "make docker-clean     - remove IoC container/network (keep image)"
	@echo "make docker-fclean    - remove IoC container/network/volumes/image"
	@echo "make setup            - prepare /tmp/ioc (venv, pip, embeddings, ollama $(LLM_MODEL))"
	@echo "make p1               - run Part 1 Overview dashboard on http://127.0.0.1:$(PORT)"
	@echo "make p2               - run Part 2 Architect API & RAG dashboard on http://127.0.0.1:$(PORT)"
	@echo "make p3               - run Part 3 Patch Loop & Dashboard on http://127.0.0.1:$(PORT)"
	@echo "make p3-cli           - run headless patch loop: make p3-cli INTENT=\"your intent\""
	@echo "make bonus            - run Chapter VII Bonus Suite & Dashboard on http://127.0.0.1:$(PORT)"
	@echo "make bonus WATCH_CONTAINER=name - same, with the Docker SDK crash watcher on that container"
	@echo "make demo-service     - run the target as a service container ($(DEMO_SERVICE)) to crash-watch"
	@echo "make demo-service-stop - remove the demo service container"
	@echo "make flake            - run flake8 on $(LINT_DIRS)"
	@echo "make mypy             - run mypy on $(LINT_DIRS)"
	@echo "make lint             - run flake8 + mypy on $(LINT_DIRS)"
	@echo "make test             - offline regression suite (installs packages + embeddings if missing; no Ollama)"
	@echo "make test-p1 / test-p2 / test-p3 / test-bonus - one part's suite only"
	@echo "make gates            - all quality gates: flake8 + mypy + test"
	@echo "make stop             - stop IoC ollama (pid file + IoC orphans; safe for make)"
	@echo "make clean            - docker-clean + remove chroma/pip caches (keep venv + weights)"
	@echo "make fclean           - docker-fclean + full wipe of /tmp/ioc"
	@echo "make re               - fclean + setup"

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
		&& [ "$(IOC_DIR)/.deps-ok" -nt "$(REQ_BONUS)" ] \
		&& $(PYTHON) -c "import chromadb,fastapi,uvicorn,watchdog,sentence_transformers,rank_bm25,httpx,yaml,docker" 2>/dev/null; then \
		echo "[*] Dependencies already ready (skip pip)"; \
	else \
		echo "[*] Installing CPU torch + Part 1, 2, 3 & bonus dependencies"; \
		$(VENV)/bin/pip install --upgrade pip setuptools wheel; \
		$(VENV)/bin/pip install --cache-dir $(PIP_CACHE) \
			torch --index-url https://download.pytorch.org/whl/cpu; \
		$(VENV)/bin/pip install --cache-dir $(PIP_CACHE) -r $(REQ) -r $(REQ_P2) -r $(REQ_P3) -r $(REQ_BONUS) pillow; \
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
# Checks BOTH the Python packages and the embedding weights: p2/p3/bonus force
# HF_HUB_OFFLINE=1, so a present venv with a missing model cache fails with a raw
# OSError traceback instead of a message that names the fix.
ensure-ready:
	@if [ ! -x "$(VENV)/bin/pip" ] \
		|| ! $(PYTHON) -c "import chromadb,fastapi,uvicorn,watchdog,sentence_transformers,rank_bm25,httpx,yaml" 2>/dev/null; then \
		echo "[!] IoC runtime not ready under $(IOC_DIR) (missing after fclean, or never set up)."; \
		echo "    1) make setup"; \
		echo "    2) make p1, make p2, make p3, or make bonus"; \
		exit 1; \
	fi
	@if [ ! -d "$(EMBED_CACHE)" ]; then \
		echo "[!] Embedding model $(EMBED_MODEL) is missing from $(HF_HOME)."; \
		echo "    The parts run fully offline, so they cannot download it on demand."; \
		echo "    Run: make setup"; \
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
	@$(EMBED_FETCH)

# What the offline test suite needs, and only that: the Python packages and the
# embedding weights, installed when missing. Never Ollama - every model call in
# the suite is a scripted fake - so `make gates` works on a fresh clone or right
# after fclean without the ollama binary or the model pull of make setup.
ensure-test-runtime: ensure-deps
	@if [ ! -d "$(EMBED_CACHE)" ]; then \
		echo "[*] Fetching embedding model $(EMBED_MODEL) into $(HF_HOME)"; \
		$(EMBED_FETCH); \
	fi

# ensure-lint-tools is part of setup because demo_app/ioc.config.yml uses
# flake8 in its validation command: the patch loop must never fail for lack of
# a tool the project itself asks for.
setup: ensure-deps ensure-ollama ensure-embed ensure-lint-tools
	@echo
	@echo "[+] Setup complete under $(IOC_DIR)"
	@echo "    Next: make p1 or p2 or p3 or bonus  →  http://127.0.0.1:$(PORT)"

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
		--ollama-host $(OLLAMA_HOST) \
		$(if $(WATCH_CONTAINER),--watch-container $(WATCH_CONTAINER))

# Chapter VII crash-watcher demo: the target app as a service container that runs
# `python3 main.py` every $(DEMO_INTERVAL) s and exits with its code on the first
# failure. The target is mounted read-only: patches land on the host copy, and the
# watcher restarts the container after a green one.
demo-service:
	@docker rm -f $(DEMO_SERVICE) >/dev/null 2>&1 || true
	@docker run -d --name $(DEMO_SERVICE) -e PYTHONDONTWRITEBYTECODE=1 \
		-v "$(abspath $(TARGET)):/srv/target:ro" -w /srv/target $(DEMO_IMAGE) \
		sh -c 'while true; do python3 main.py || exit $$?; sleep $(DEMO_INTERVAL); done' >/dev/null
	@echo "[+] $(DEMO_SERVICE) runs $(TARGET)/main.py every $(DEMO_INTERVAL)s (logs: docker logs -f $(DEMO_SERVICE))"
	@echo "    Watch it: make bonus WATCH_CONTAINER=$(DEMO_SERVICE)  (or the Crash Watcher card)"

demo-service-stop:
	@docker rm -f $(DEMO_SERVICE) >/dev/null 2>&1 && echo "[+] $(DEMO_SERVICE) removed" \
		|| echo "[*] $(DEMO_SERVICE) was not running"

stop:
	@# Prefer pid file written by ensure-ollama*
	@if [ -f $(IOC_DIR)/ollama.pid ]; then \
		kill $$(cat $(IOC_DIR)/ollama.pid) 2>/dev/null || true; \
		rm -f $(IOC_DIR)/ollama.pid; \
		echo "[*] Stopped ollama via $(IOC_DIR)/ollama.pid"; \
	fi
	@# Orphans after lost pid: only ollama whose environ uses IoC models dir.
	@# Never pkill -f MODELS/HOST strings - that matches this recipe and kills make.
	@for pid in $$(pgrep -x ollama 2>/dev/null || true); do \
		if [ -r /proc/$$pid/environ ] \
			&& tr '\0' '\n' < /proc/$$pid/environ 2>/dev/null \
				| grep -qx 'OLLAMA_MODELS=$(OLLAMA_DIR)'; then \
			kill $$pid 2>/dev/null || true; \
			echo "[*] Stopped orphan ollama pid $$pid"; \
		fi; \
	done
	@rm -f $(IOC_DIR)/ollama.pid 2>/dev/null || true
	@echo "[*] stop done"

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

# ---------------------------------------------------------------------------
# Tests + quality gates
# ---------------------------------------------------------------------------

# Offline regression suite: p1/tests, p2/tests, p3/tests, bonus/tests. Plain
# stdlib unittest on the real ChromaDB and embedding model, which
# ensure-test-runtime installs when they are missing; never Ollama. Targets are
# temp copies of demo_app, so the real one is never touched. Narrow it with
# e.g. TEST_ARGS="-k rollback".
test: ensure-test-runtime
	@echo "[*] unittest → $(TEST_DIRS)"
	@HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m unittest discover \
		-s . -t . -p "test_*.py" $(TEST_ARGS)

# One part's suite only: make test-p1 / test-p2 / test-p3 / test-bonus.
test-p1 test-p2 test-p3 test-bonus: test-%: ensure-test-runtime
	@echo "[*] unittest → $*/tests"
	@HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m unittest discover \
		-s $*/tests -t . -p "test_*.py" $(TEST_ARGS)

# Every quality gate, in order: style (flake8), types (mypy), behaviour (unittest).
gates: lint test
	@echo "[+] all gates green (flake8 + mypy + unittest)"

# Soft clean: stop ollama + docker container/network + local caches.
# Keeps the venv AND the embedding weights: deleting $(HF_HOME) left every part
# broken (they run offline and cannot refetch it), which is not what a soft
# clean should do. Use make fclean for a full wipe.
clean: stop docker-clean
	@rm -rf $(CHROMA_DIR) $(PIP_CACHE) $(IOC_DIR)/logs
	@echo "[*] Cleaned chroma db, pip cache and logs under $(IOC_DIR)"
	@echo "    (venv and embedding weights kept - make p1/p2/p3/bonus still work)"

# Full clean: stop ollama + remove IoC docker image/network/volumes + wipe /tmp/ioc
fclean: stop docker-fclean
	@rm -rf $(IOC_DIR)
	@echo "[*] Removed $(IOC_DIR)"

re: fclean setup
