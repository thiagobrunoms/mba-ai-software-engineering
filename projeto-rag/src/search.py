import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

# PGVector retorna distância cosseno: 0 = idêntico, 2 = oposto.
# Documentos com score acima desse limiar são considerados irrelevantes.
SCORE_THRESHOLD = 0.5
NO_KNOWLEDGE_MSG = "Não tenho informações necessárias para responder sua pergunta."


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
    def openai_embedding_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "text-embedding-3-small")

    @property
    def openai_chat_model(self) -> str:
        return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


class VectorStore:
    def __init__(self, config: Config):
        embeddings = OpenAIEmbeddings(model=config.openai_embedding_model)
        self._store = PGVector(
            embeddings=embeddings,
            collection_name=config.pgvector_collection,
            connection=config.pgvector_url,
            use_jsonb=True,
        )

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k)


class Retriever:
    """Busca documentos relevantes e filtra pelo score de distância cosseno."""

    def __init__(self, store: VectorStore, score_threshold: float = SCORE_THRESHOLD, k: int = 10):
        self._store = store
        self._threshold = score_threshold
        self._k = k

    def retrieve(self, query: str) -> str:
        results = self._store.search(query, k=self._k)
        relevant = [doc for doc, score in results if score <= self._threshold]
        if not relevant:
            return NO_KNOWLEDGE_MSG
        return "\n\n---\n\n".join(doc.page_content.strip() for doc in relevant)


class RAGChain:
    """Chain LCEL: recupera contexto do VectorStore → responde com LLM.

    Se o contexto não for encontrado, retorna NO_KNOWLEDGE_MSG diretamente
    sem chamar o LLM, evitando alucinações.
    """

    _PROMPT = ChatPromptTemplate.from_template(
        "CONTEXTO:\n{context}\n\n"
        "REGRAS:\n"
        "- Responda somente com base no CONTEXTO.\n"
        "- Se a informação não estiver explicitamente no CONTEXTO, responda:\n"
        '  "Não tenho informações necessárias para responder sua pergunta."\n'
        "- Nunca invente ou use conhecimento externo.\n"
        "- Nunca produza opiniões ou interpretações além do que está escrito.\n\n"
        "EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:\n"
        'Pergunta: "Qual é a capital da França?"\n'
        'Resposta: "Não tenho informações necessárias para responder sua pergunta."\n\n'
        'Pergunta: "Quantos clientes temos em 2024?"\n'
        'Resposta: "Não tenho informações necessárias para responder sua pergunta."\n\n'
        'Pergunta: "Você acha isso bom ou ruim?"\n'
        'Resposta: "Não tenho informações necessárias para responder sua pergunta."\n\n'
        "PERGUNTA DO USUÁRIO:\n{question}\n\n"
        'RESPONDA A "PERGUNTA DO USUÁRIO"'
    )

    def __init__(self, config: Config):
        store = VectorStore(config)
        self._retriever = Retriever(store)
        self._chain = self._PROMPT | ChatOpenAI(model=config.openai_chat_model, temperature=0) | StrOutputParser()

    def run(self, query: str) -> str:
        context = self._retriever.retrieve(query)
        if context == NO_KNOWLEDGE_MSG:
            return NO_KNOWLEDGE_MSG

        return self._chain.invoke({
            "context": context,
            "question": query,
            "no_knowledge_msg": NO_KNOWLEDGE_MSG,
        })


if __name__ == "__main__":
    config = Config()
    chain = RAGChain(config)

    queries = [
        "Quais contrapesos a antiga disciplina estoica oferece?",
        "Qual a capital da França?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Pergunta: {query}")
        print(f"{'='*60}")
        print(chain.run(query))
