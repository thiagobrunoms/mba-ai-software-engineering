# RAG Project

RAG (Retrieval-Augmented Generation) using LangChain, PGVector, and OpenAI. The system ingests PDF documents into a vector database and answers questions based solely on the ingested content, avoiding hallucinations.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key

## Setup

### 1. Start the database

```bash
docker compose up -d
```

### 2. Configure environment variables

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_MODEL=text-embedding-3-small
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=gpt5_collection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Document Ingestion

Place the PDF at the project root as `document.pdf` and run:

```bash
python src/ingest.py
```

This loads the PDF, splits it into chunks of 1000 characters (overlap 150), generates embeddings, and stores them in PGVector.

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

Você: o que é estoicismo?

Assistente: O estoicismo é uma filosofia prática, criada para orientar a vida humana em meio à instabilidade, ao sofrimento e à incerteza. Ele propõe um modo de atravessar perdas, desejos, medos, conflitos e responsabilidades com lucidez, disciplina e senso de realidade. O estoicismo não promete uma existência sem dor, mas ensina a viver de acordo com a natureza, aceitando a impermanência e agindo com integridade. Além disso, enfatiza a importância do caráter, da justiça e da lucidez, promovendo uma vida de ordem interior em meio aos problemas.

Você: Quantos clientes temos em 2024?

Assistente: Não tenho informações necessárias para responder sua pergunta.
```

> type "sair" to finish the chat.

> If a question has no answer in the ingested documents, the system will say so instead of making one up.
