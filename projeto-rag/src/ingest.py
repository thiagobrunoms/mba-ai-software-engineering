import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector


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


class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=False,
        )

    def load(self, pdf_path: Path) -> list[Document]:
        raw_docs = PyPDFLoader(str(pdf_path)).load()
        splits = self.splitter.split_documents(raw_docs)
        if not splits:
            raise SystemExit(0)
        return [self._enrich(doc) for doc in splits]

    @staticmethod
    def _enrich(doc: Document) -> Document:
        return Document(
            page_content=doc.page_content,
            metadata={k: v for k, v in doc.metadata.items() if v not in ("", None)},
        )


class VectorStore:
    def __init__(self, config: Config):
        embeddings = OpenAIEmbeddings(model=config.openai_model)
        self._store = PGVector(
            embeddings=embeddings,
            collection_name=config.pgvector_collection,
            connection=config.pgvector_url,
            use_jsonb=True,
        )

    def save(self, documents: list[Document]) -> None:
        ids = [f"docx-{i}" for i in range(len(documents))]
        self._store.add_documents(documents=documents, ids=ids)


class RAGInjector:
    def __init__(self, pdf_path: Path):
        self.config = Config()
        self.loader = DocumentLoader()
        self.store = VectorStore(self.config)
        self.pdf_path = pdf_path

    def run(self) -> None:
        documents = self.loader.load(self.pdf_path)
        self.store.save(documents)


if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent / "document.pdf"
    RAGInjector(pdf_path).run()
