import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document


class Config:
    REQUIRED_VARS = ("PGVECTOR_URL", "PGVECTOR_COLLECTION")

    def __init__(self):
        load_dotenv()
        self._validate()

    def _validate(self):
        for key in self.REQUIRED_VARS:
            if not os.getenv(key):
                raise RuntimeError(f"Environment variable {key} is not set")

    @property
    def pgvector_url(self) -> str:
        return os.getenv("PGVECTOR_URL")

    @property
    def pgvector_collection(self) -> str:
        return os.getenv("PGVECTOR_COLLECTION")

    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "text-embedding-3-small")


class VectorStore:
    def __init__(self, config: Config):
        embeddings = OpenAIEmbeddings(model=config.openai_model)
        self._store = PGVector(
            embeddings=embeddings,
            collection_name=config.pgvector_collection,
            connection=config.pgvector_url,
            use_jsonb=True,
        )

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k)


class ResultPrinter:
    def print(self, results: list[tuple[Document, float]]) -> None:
        for i, (doc, score) in enumerate(results, start=1):
            print("\n\n = INIT =")
            print(f"Resultado {i} (score: {score:.2f}):")
            print("\nTexto:\n")
            print(doc.page_content.strip())
            print("\nMetadados:\n")
            for k, v in doc.metadata.items():
                print(f"{k}: {v}")


class RAGSearch:
    def __init__(self, query: str, k: int = 3):
        self.config = Config()
        self.store = VectorStore(self.config)
        self.printer = ResultPrinter()
        self.query = query
        self.k = k

    def run(self) -> None:
        results = self.store.search(self.query, k=self.k)
        self.printer.print(results)


if __name__ == "__main__":
    query = "Quais contrapesos a antiga disciplina estoica oferece?"
    RAGSearch(query).run()
