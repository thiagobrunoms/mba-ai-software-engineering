# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the scripts

All scripts must be run from the project root with the virtualenv active:

```bash
source venv/bin/activate

# Ingest PDF into vector database
python src/ingest.py

# Run predefined queries against the vector store
python src/search.py

# Start interactive chat
python src/chat.py
```

## Required environment variables

Create a `.env` file at the project root:

```
PGVECTOR_URL=postgresql://user:password@localhost:5432/dbname
PGVECTOR_COLLECTION=collection_name
OPENAI_API_KEY=sk-...
OPENAI_MODEL=text-embedding-3-small        # optional, embedding model
OPENAI_CHAT_MODEL=gpt-4o-mini              # optional, chat/LLM model
```

## Architecture

The project is a RAG pipeline split into two stages:

**Ingestion** (`src/ingest.py`): Loads `estoicismo.pdf` (project root) → splits into chunks via `RecursiveCharacterTextSplitter` → generates embeddings with `OpenAIEmbeddings` → stores in PGVector.

**Retrieval + Generation** (`src/search.py`): Given a query, fetches top-k similar chunks from PGVector using cosine distance. Chunks with score above `SCORE_THRESHOLD = 0.5` are discarded. If no relevant chunk remains, returns a fixed "no knowledge" message without calling the LLM, avoiding hallucinations. Otherwise, passes the context to a `ChatOpenAI` chain via LCEL (`prompt | llm | parser`).

**Chat interface** (`src/chat.py`): Thin REPL wrapper around `RAGChain` from `search.py`. Exit keyword: `sair`.

Both `ingest.py` and `search.py` define their own `Config` and `VectorStore` classes independently — they are not shared modules.
