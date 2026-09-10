*This project has been created as part of the 42 curriculum by ajovanov, iberegsz.*

# Inception-of-Context

## Description

Inception-of-Context (IoC) is a local, offline codebase intelligence toolchain.
**Part 1** turns a target project into a synchronized vector index:

- walks a codebase (default: `demo_app/`) with ignore rules for caches, VCS, and the DB folder itself
- splits source into **logical AST chunks** (functions, classes, methods)
- embeds chunks locally with **all-MiniLM-L6-v2** and stores them in **ChromaDB**
- watches the filesystem (debounce + polling fallback) so the index stays in sync
- serves an **Overview dashboard** (FastAPI + SSE) on `http://127.0.0.1:8000`

All runtime state (venv, pip cache, embedding weights, ChromaDB, Ollama working files) lives under **`/tmp/ioc`** so the home directory stays small.

### Repository layout

```text
inception-of-context/
├── p1/                      # Part 1 implementation
│   ├── index.py
│   ├── indexer.py
│   ├── chunker.py
│   ├── db.py
│   ├── watcher.py
│   ├── dashboard.py
│   └── requirements.txt
├── demo_app/                # Target codebase for indexing
│   ├── calculator.py
│   ├── formatter.py
│   └── main.py
├── presentation/            # Slide deck
│   ├── PRESENTATION.html    # English
│   └── PRESENTATION_HU.html # Hungarian
├── Makefile                 # setup, p1, clean, fclean → /tmp/ioc
├── README.md
└── .gitignore
```

## Instructions

### Requirements

- Python 3.10+
- `virtualenv` (preferred if `python3 -m venv` / `ensurepip` is missing)
- `ollama` (for the local LLM label / Part-ready model `qwen2.5:3b`)
- Network once for pip packages, the embedding model, and `ollama pull`

### Quick start (Makefile)

From the repository root:

```bash
make setup   # once (or after reboot / make clean)
make p1      # start Part 1 dashboard
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). Stop with `Ctrl+C`.

| Target | What it does |
|--------|----------------|
| `make setup` | Creates `/tmp/ioc`, installs the venv + deps, pulls `qwen2.5:3b`, prefetches embeddings |
| `make p1` | Part 1  indexes `demo_app` and launches the Overview dashboard on port 8000 |
| `make p2` | Stub  Part 2 Ask & Retrieve (not implemented yet) |
| `make p3` | Stub  Part 3 Patch Loop (not implemented yet) |
| `make bonus` | Stub  bonus features (not implemented yet) |
| `make clean` | Removes chroma DB + pip/HF caches (keeps venv) |
| `make fclean` | Full wipe of `/tmp/ioc` |
| `make re` | `fclean` then `setup` |
| `make stop` | Stops an `ollama serve` process started by this Makefile |

After reboot, `/tmp` is empty  run `make setup` again.

### Manual equivalent (same layout)

```bash
mkdir -p /tmp/ioc/{venv,pip-cache,hf-cache,chroma_db}

# venv (use virtualenv on campus images without ensurepip)
virtualenv /tmp/ioc/venv

/tmp/ioc/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/ioc/venv/bin/pip install -r p1/requirements.txt pillow

# LLM
ollama serve &          # if not already running
ollama pull qwen2.5:3b

# run dashboard (one short env block)
export PYTHONNOUSERSITE=1
export PYTHONPATH=/tmp/ioc/venv/lib/python3.10/site-packages
export HF_HOME=/tmp/ioc/hf-cache
export TMPDIR=/tmp

python3 p1/index.py demo_app \
  --db-dir /tmp/ioc/chroma_db \
  --watch --dashboard --port 8000 \
  --llm-model qwen2.5:3b
```

Or, after `make setup`, use the generated helper:

```bash
/tmp/ioc/run.sh demo_app --db-dir /tmp/ioc/chroma_db --watch --dashboard --port 8000
```

Do **not** commit `.venv`, `.chroma_db`, or model weights  they are gitignored and belong under `/tmp/ioc`.

## Resources

- [ChromaDB documentation](https://docs.trychroma.com/)
- [sentence-transformers  pretrained models](https://www.sbert.net/docs/pretrained_models.html) (`all-MiniLM-L6-v2`)
- [Python `ast` module](https://docs.python.org/3/library/ast.html)
- [watchdog  filesystem events](https://python-watchdog.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/) / [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [Ollama](https://ollama.com/) / [Ollama model library](https://ollama.com/library)
- [Hugging Face  sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
