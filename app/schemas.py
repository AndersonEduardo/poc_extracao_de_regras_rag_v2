from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class RAGRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Pergunta do usuário para o sistema RAG.",
    )


class RetrievedSource(BaseModel):
    rank: int
    content_preview: str
    metadata: dict[str, Any]

class RagAnswerItem(BaseModel):
    suggested_items: list[str] = Field(alias="suggested-items")
    clinical_rationale: str = Field(alias="clinical-rationale")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

class RagAnswer(BaseModel):
    status: str
    results: list[RagAnswerItem]

    model_config = ConfigDict(extra="forbid")

class RAGResponse(BaseModel):
    query: str
    answer: RagAnswer
    sources: list[RetrievedSource]
    num_documents_retrieved: int
