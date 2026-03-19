# RAG Project

RAG (Retrieval-Augmented Generation) using LangChain, PGVector, and OpenAI. The system ingests PDF documents into a vector database and answers questions based solely on the ingested content, avoiding hallucinations.

## Prerequisites

- Python 3.11+
- PostgreSQL with the `pgvector` extension
- OpenAI API key

## Setup

Create a `.env` file at the project root:

```env
PGVECTOR_URL=postgresql://user:password@localhost:5432/dbname
PGVECTOR_COLLECTION=collection_name
OPENAI_API_KEY=sk-...
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Document Ingestion

Place the desired PDF in `src/` and run:

```bash
python src/ingest.py
```

This loads the PDF, splits it into chunks, and stores the embeddings in PGVector.

## search.py

Runs one-off searches against the vector database using predefined questions. Useful for quick tests or validating ingested content.

```bash
python src/search.py
```

The questions are defined directly in the `__main__` block. To change them, edit the `queries` list in [src/search.py](src/search.py).

## chat.py

Interactive chat interface in the terminal. The user types questions freely and receives answers based on the ingested documents.

```bash
python src/chat.py
```

- Type your question and press Enter to get an answer.
- To exit, type `sair`.

**Example usage:**

```
Chat RAG iniciado. Digite 'sair' para encerrar.

Você: Quais contrapesos a antiga disciplina estoica oferece?
Assistente: ...

Você: sair
Encerrando chat. Até logo!
```

> If a question has no answer in the ingested documents, the system will say so instead of making one up.
