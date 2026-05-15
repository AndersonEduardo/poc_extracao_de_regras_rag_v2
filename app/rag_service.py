import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.prompts import *
from app.schemas import RagAnswer


class RAGService:
    """
    Serviço simples de RAG usando:
    - ChromaDB já persistido com documentos indexados;
    - OpenAIEmbeddings para vetorizar a query;
    - ChatOpenAI para gerar a resposta final.

    Esta classe foi pensada para ser reutilizada dentro de uma API FastAPI.
    """

    def __init__(
        self,
        persist_directory:str,
        collection_name:str,
        openai_chat_model:str,
        openai_embedding_model:str,
        top_k:int,
        temperature:float,
    ) -> None:
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.top_k = top_k

        self.embeddings = OpenAIEmbeddings(
            model = openai_embedding_model,
        )

        self.vector_store = Chroma(
            collection_name = collection_name,
            persist_directory = persist_directory,
            embedding_function = self.embeddings,
        )

        self.llm = ChatOpenAI(
            model = openai_chat_model, 
            temperature = temperature,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", QUERY_PROMPT),
            ]
        )


    def retrieve(self, query: str) -> list[Document]:
        """
        Recupera os documentos mais relevantes do ChromaDB.
        """

        return self.vector_store.similarity_search(
            query = query,
            k = self.top_k,
        )


    def format_context(self, documents:list[Document]) -> str:
        """
        Converte os documentos recuperados em uma string única para o prompt.
        """

        if not documents:

            return "Nenhum contexto foi recuperado."

        documents = sorted(documents, key=lambda x: x.metadata['chunk_index'])

        context_parts = []

        for idx, doc in enumerate(documents, start=1):

            source = doc.metadata.get("source", "fonte_desconhecida")

            context_parts.append(
                f"[Documento {idx}]\n"
                f"Chunk index: {doc.metadata['chunk_index']}\n"
                f"Fonte: {source}\n"
                f"Conteúdo:\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)


    def answer(self, query: str) -> dict[str, Any]:
        """
        Executa o fluxo completo:
        1. Recebe a pergunta;
        2. Faz retrieval no ChromaDB;
        3. Monta o contexto;
        4. Chama a LLM;
        5. Retorna resposta + fontes.
        """

        documents = self.retrieve(query)
        context = self.format_context(documents)

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": query,
                "context": context,
            }
        )

        raw_answer = response.content

        answer = RagAnswer.model_validate_json(raw_answer)

        sources = self.extract_sources(documents)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "num_documents_retrieved": len(documents),
        }


    def extract_sources(self, documents: list[Document]) -> list[dict[str, Any]]:
        """
        Extrai metadados úteis dos documentos recuperados.
        """

        sources = []

        for idx, doc in enumerate(documents, start=1):

            sources.append(
                {
                    "rank": idx,
                    "content_preview": doc.page_content, #[:500], # manter texto completo do chunk.
                    "metadata": doc.metadata,
                }
            )

        return sources