import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException

from app.rag_service import RAGService
from app.schemas import RAGRequest, RAGResponse


load_dotenv()

app = FastAPI(
    title       = "RAG API",
    description = "API simples para Retrieval Augmented Generation usando LangChain, OpenAI e ChromaDB.",
    version     = "0.0.1",
)


@lru_cache
def get_rag_service() -> RAGService:
    """
    Cria uma única instância cacheada do RAGService.

    Isso evita reconstruir embeddings, vector store e LLM a cada request.
    """

    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY")
    collection_name = os.getenv('CHROMA_COLLECTION_NAME')

    openai_chat_model = os.getenv('OPENAI_CHAT_MODEL')
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")

    top_k = int(os.getenv("RAG_TOP_K"))

    return RAGService(
        persist_directory       = persist_directory,
        collection_name         = collection_name,
        openai_chat_model       = openai_chat_model,
        openai_embedding_model  = openai_embedding_model,
        top_k                   = top_k,
        temperature             = 0.0,
    )


@app.get("/health")
def health_check() -> dict[str, str]:

    return {"status": "ok"}


@app.post("/rag/answer", response_model=RAGResponse)
def answer_question(
    payload: RAGRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """
    Endpoint principal do RAG.

    Recebe uma query do usuário, faz retrieval no ChromaDB
    e retorna a resposta gerada pela LLM.
    """

    try:

        result = rag_service.answer(payload.query)
        
        return RAGResponse(**result)

    except Exception as exc:

        raise HTTPException(
            status_code = 500,
            detail = f"Erro ao processar a pergunta: {str(exc)}",
        ) from exc
