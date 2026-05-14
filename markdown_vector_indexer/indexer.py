from __future__ import annotations

import argparse
import json
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import chromadb
import yaml
from dotenv import load_dotenv
from openai import OpenAI

LOGGER = logging.getLogger(__name__)

FRONT_MATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n*", re.DOTALL)
SECTION_PATTERN = re.compile(
    r"(?=^#{1,6}\s+.+$|^<!-- page-break -->$)",
    re.MULTILINE,
)
SECTION_HEADING_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")
KEYWORDS_EXTRACTION_PROMPT = '''
# SISTEMA

Você é um especialista em terminologia médica e extração estruturada de informação clínica.

# TAREFA

Analise o texto de guideline médico fornecido e extraia todos os termos médicos explicitamente presentes no texto, produzindo um dicionário JSON normalizado.

# INSTRÇÕES

1. Leia todo o texto passo a passo e com atenção.
2. Identifique apenas termos médicos explicitamente mencionados.
3. Considere como termos médicos, quando aparecerem no texto:
   - doenças, síndromes e condições clínicas;
   - sinais, sintomas e achados clínicos;
   - exames, procedimentos e intervenções;
   - medicamentos e classes terapêuticas;
   - estruturas anatômicas;
   - escalas, escores, classificações e sistemas de codificação médica.
4. Normalize variações ortográficas, tipográficas e de formatação em uma única chave canônica.
   Exemplo: "CID10", "CID-10", "cid 10" e "diagnóstico cid-10" -> "cid10".
5. Para cada termo normalizado, escreva uma explicação breve, objetiva e clara.
6. Não invente termos não presentes no texto.
7. Não inclua palavras genéricas, administrativas ou não médicas.
8. Quando duas expressões forem claramente equivalentes no contexto, consolide-as em uma única chave.
9. Retorne apenas JSON válido, sem comentários, sem markdown e sem texto adicional.
10. Adcione ao dicionário todos os códigos CID-10 mencinados  no texto, com suas respectivas explicações.
11. Adicione ao dicionário códigos CID-10 para todas as condições clínicas mencionadas no texto, mesmo que o código específico não seja mencionado, usando seu conhecimento médico para inferir o código mais apropriado. Por exemplo:
    11.1. Se o texto mencionar "paciente com diabetes", adicione a chave "cid10": "e10, e11, e13 ou e14" com a explicação "Código CID-10 para diabetes mellitus, incluindo tipos 1, 2, outros especificados e não especificados."
    11.2. Se o texto mencionar "paciente com hipertensão", adicione a chave "cid10": "i10" com a explicação "Código CID-10 para hipertensão essencial (primária)."
    11.3. Se o texto mencionar "paciente relata histórico de depressão", adicione a chave "cid10": "f32, f33 ou f34.1" com a explicação "Códigos CID-10 para episódios depressivos, transtorno depressivo recorrente e distimia."
    Etc.

# REGRAS DE NORMALIZAÇÃO

1. usar minúsculas;
2. remover espaços extras;
3. remover hífens quando não alterarem o significado;
4. padronizar siglas e variantes gráficas;
5. manter termos diferentes separados quando tiverem significados clínicos distintos.

Formato de saída:
{
  "termo_normalizado": "explicação breve do significado"
}
'''

CHUNK_SUMMARY_PROMPT = '''
# SISTEMA

Você é um especialista em sumarização clínica para recuperação semântica de documentos.

# TAREFA

Leia o trecho de documento clínico fornecido e produza um resumo curto, específico e informativo sobre o contexto daquele chunk no documento principal.

# REGRAS

1. Escreva exatamente uma frase curta e informativa.
2. Foque no assunto clínico, critérios, condutas, exames, medicamentos ou população abordada no trecho.
3. Use apenas informações sustentadas pelo trecho e pelo contexto mínimo do documento.
4. Não invente recomendações, critérios ou diagnósticos ausentes.
5. Retorne apenas JSON válido, sem markdown e sem texto adicional.

Formato de saída:
{
  "chunk_summary": "Trecho sobre ..."
}
'''

SECTION_SUMMARY_PROMPT = '''
# SISTEMA

Você é um especialista em sumarização clínica para recuperação semântica de documentos.

# TAREFA

Leia uma seção de documento clínico em Markdown e produza uma frase curta, específica e informativa que descreva o conteúdo da seção no documento principal.

# REGRAS

1. Escreva exatamente uma frase curta e informativa.
2. Foque no tema clínico, critérios, condutas, exames, medicamentos ou população abordada na seção.
3. Use apenas informações sustentadas pelo título e pelo conteúdo da seção.
4. Não invente recomendações, critérios ou diagnósticos ausentes.
5. Retorne apenas JSON válido, sem markdown e sem texto adicional.

Formato de saída:
{
  "section_summary": "Seção sobre ..."
}
'''



@dataclass(slots=True)
class ChunkingConfig:
    max_characters: int = 2_200 # 1_200
    overlap_characters: int = 400 # 200
    min_characters: int = 150


@dataclass(slots=True)
class OpenAIEmbeddingConfig:
    model: str = "text-embedding-3-small"
    batch_size: int = 32


@dataclass(slots=True)
class IndexingConfig:
    input_dir: Path
    chroma_dir: Path
    keywords_output_dir: Path
    collection_name: str = "markdown_documents"
    recursive: bool = True
    reset_collection: bool = False
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: OpenAIEmbeddingConfig = field(default_factory=OpenAIEmbeddingConfig)


@dataclass(slots=True)
class IndexingSummary:
    total_documents: int
    total_chunks: int
    collection_name: str
    chroma_dir: Path


@dataclass(slots=True)
class MarkdownDocument:
    source_path: Path
    relative_path: Path
    metadata: dict[str, object]
    body: str


@dataclass(slots=True)
class MarkdownChunk:
    chunk_id: str
    text: str
    metadata: dict[str, object]


@dataclass(slots=True)
class MarkdownSection:
    section_index: int
    title: str
    text: str


class MarkdownChunker:
    """Segmenta markdown priorizando secoes e aplicando overlap por caracteres."""

    def __init__(self, config: ChunkingConfig) -> None:
        if config.overlap_characters >= config.max_characters:
            raise ValueError("chunk_overlap deve ser menor que chunk_size")
        self.config = config

    def chunk_document(self, document: MarkdownDocument) -> list[MarkdownChunk]:
        chunks: list[MarkdownChunk] = []
        for section in self._split_document_sections(document.body):
            section_parts = self._split_sections(section.text)
            for text in self._build_raw_chunks(section_parts):
                chunk_index = len(chunks)
                chunks.append(
                    MarkdownChunk(
                        chunk_id=self._make_chunk_id(document.relative_path, chunk_index, text),
                        text=text,
                        metadata={
                            **document.metadata,
                            "source_path": str(document.source_path),
                            "relative_path": str(document.relative_path),
                            "chunk_index": chunk_index,
                            "chunk_size": len(text),
                            "section_index": section.section_index,
                            "section_title": section.title,
                        },
                    )
                )
        return chunks

    def _split_document_sections(self, text: str) -> list[MarkdownSection]:
        matches = list(SECTION_HEADING_PATTERN.finditer(text))
        sections: list[MarkdownSection] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue
            sections.append(
                MarkdownSection(
                    section_index=len(sections),
                    title=match.group(1).strip(),
                    text=section_text,
                )
            )
        return sections

    def _split_sections(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        sections = [
            section.strip()
            for section in SECTION_PATTERN.split(normalized)
            if section.strip()
        ]
        return sections or [normalized]

    def _build_raw_chunks(self, sections: list[str]) -> list[str]:
        chunks: list[str] = []
        buffer = ""

        for section in sections:
            candidate = self._join_parts(buffer, section)
            if len(candidate) <= self.config.max_characters:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = ""

            if len(section) <= self.config.max_characters:
                buffer = section
                continue

            chunks.extend(self._split_large_section(section))

        if buffer:
            chunks.append(buffer)

        return self._merge_small_tail_chunks(chunks)

    def _split_large_section(self, section: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip()]
        chunks: list[str] = []
        buffer = ""

        for paragraph in paragraphs:
            candidate = self._join_parts(buffer, paragraph)
            if len(candidate) <= self.config.max_characters:
                buffer = candidate
                continue

            if buffer:
                chunks.append(buffer)
                buffer = self._overlap_tail(buffer)

            while len(paragraph) > self.config.max_characters:
                head = paragraph[: self.config.max_characters].strip()
                if head:
                    chunks.append(head)
                paragraph = paragraph[
                    self.config.max_characters - self.config.overlap_characters :
                ].strip()

            buffer = self._join_parts(buffer, paragraph)

        if buffer:
            chunks.append(buffer)

        return chunks

    def _merge_small_tail_chunks(self, chunks: list[str]) -> list[str]:
        if len(chunks) < 2:
            return chunks

        merged: list[str] = []
        for chunk in chunks:
            if (
                merged
                and len(chunk) < self.config.min_characters
                and len(self._join_parts(merged[-1], chunk)) <= self.config.max_characters
            ):
                merged[-1] = self._join_parts(merged[-1], chunk)
                continue
            merged.append(chunk)
        return merged

    def _overlap_tail(self, text: str) -> str:
        tail = text[-self.config.overlap_characters :].strip()
        return tail

    @staticmethod
    def _join_parts(left: str, right: str) -> str:
        if not left:
            return right.strip()
        if not right:
            return left.strip()
        return f"{left.rstrip()}\n\n{right.lstrip()}".strip()

    @staticmethod
    def _make_chunk_id(relative_path: Path, index: int, text: str) -> str:
        digest = hashlib.sha1(f"{relative_path}:{index}:{text}".encode("utf-8")).hexdigest()
        return digest


class MarkdownVectorIndexer:
    """Carrega markdowns, gera embeddings e persiste os chunks em um ChromaDB."""

    def __init__(self, config: IndexingConfig) -> None:
        self.config = config
        self.chunker = MarkdownChunker(config.chunking)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY nao encontrada. Defina a chave no arquivo .env do projeto."
            )
        config.chroma_dir.mkdir(parents=True, exist_ok=True)
        config.keywords_output_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path=str(config.chroma_dir))

    def index_documents(self) -> IndexingSummary:
        documents = self._load_documents()
        for document in documents:
            keywords = self._extract_keywords(document)
            self._save_keywords(document, keywords)
        chunks: list[MarkdownChunk] = []
        for document in documents:
            document_chunks = self.chunker.chunk_document(document)
            document_chunks = self._add_section_summaries(document, document_chunks)
            chunks.extend(self._add_chunk_summaries(document, document_chunks))

        collection = self._get_collection()
        self._upsert_chunks(collection, chunks)

        return IndexingSummary(
            total_documents=len(documents),
            total_chunks=len(chunks),
            collection_name=self.config.collection_name,
            chroma_dir=self.config.chroma_dir,
        )

    def _load_documents(self) -> list[MarkdownDocument]:
        input_dir = self.config.input_dir.expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Diretorio de entrada nao existe: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"Caminho de entrada nao e um diretorio: {input_dir}"
            )

        pattern = "**/*.md" if self.config.recursive else "*.md"
        markdown_files = sorted(input_dir.glob(pattern))
        LOGGER.info("Encontrados %s arquivos markdown em %s", len(markdown_files), input_dir)

        return [self._read_markdown(path, input_dir) for path in markdown_files]

    def _read_markdown(self, path: Path, root_dir: Path) -> MarkdownDocument:
        content = path.read_text(encoding="utf-8")
        metadata, body = self._split_front_matter(content)
        metadata.setdefault("document_name", path.stem)
        metadata.setdefault("document_extension", path.suffix)

        return MarkdownDocument(
            source_path=path.resolve(),
            relative_path=path.resolve().relative_to(root_dir),
            metadata=self._sanitize_metadata(metadata),
            body=body.strip(),
        )

    def _get_collection(self):
        name = self.config.collection_name
        if self.config.reset_collection:
            try:
                self.chroma.delete_collection(name)
                LOGGER.info("Colecao %s removida antes da reindexacao", name)
            except Exception:
                LOGGER.debug("Colecao %s nao existia antes do reset", name)

        return self.chroma.get_or_create_collection(name=name)

    def _upsert_chunks(self, collection, chunks: list[MarkdownChunk]) -> None:
        if not chunks:
            LOGGER.warning("Nenhum chunk foi gerado. Nada para indexar.")
            return

        batch_size = self.config.embedding.batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            raw_texts = [chunk.text for chunk in batch]
            embedding_texts = [self._build_embedding_text(chunk) for chunk in batch]
            embeddings = self._embed_texts(embedding_texts)
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=raw_texts,
                metadatas=[chunk.metadata for chunk in batch],
                embeddings=embeddings,
            )
            LOGGER.info(
                "Persistidos chunks %s-%s de %s",
                start,
                start + len(batch) - 1,
                len(chunks),
            )

    def _add_section_summaries(
        self,
        document: MarkdownDocument,
        chunks: list[MarkdownChunk],
    ) -> list[MarkdownChunk]:
        section_summaries: dict[tuple[object, object], str] = {}
        summarized_chunks: list[MarkdownChunk] = []
        for chunk in chunks:
            section_key = (
                chunk.metadata.get("section_index"),
                chunk.metadata.get("section_title"),
            )
            if section_key not in section_summaries:
                section_text = self._section_text_from_chunks(chunks, section_key)
                section_summaries[section_key] = self._summarize_section(
                    document,
                    str(chunk.metadata.get("section_title", "")),
                    section_text,
                )
            summarized_chunks.append(
                MarkdownChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "section_summary": section_summaries[section_key],
                    },
                )
            )
        return summarized_chunks

    @staticmethod
    def _section_text_from_chunks(
        chunks: list[MarkdownChunk],
        section_key: tuple[object, object],
    ) -> str:
        return "\n\n".join(
            chunk.text
            for chunk in chunks
            if (
                chunk.metadata.get("section_index"),
                chunk.metadata.get("section_title"),
            )
            == section_key
        )

    def _summarize_section(
        self,
        document: MarkdownDocument,
        section_title: str,
        section_text: str,
    ) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SECTION_SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": self._build_section_summary_user_prompt(
                        document,
                        section_title,
                        section_text,
                    ),
                },
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError(
                f"A OpenAI retornou uma resposta vazia para o resumo da seção {section_title}."
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"A resposta da LLM para resumo da seção {section_title} nao veio em JSON valido."
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("section_summary"), str):
            raise ValueError(
                f"A resposta da LLM para resumo da seção {section_title} deve conter section_summary string."
            )

        summary = payload["section_summary"].strip()
        if not summary:
            raise ValueError(
                f"A resposta da LLM para resumo da seção {section_title} retornou section_summary vazio."
            )
        return summary

    @staticmethod
    def _build_section_summary_user_prompt(
        document: MarkdownDocument,
        section_title: str,
        section_text: str,
    ) -> str:
        return "\n".join(
            [
                f"document_name={document.metadata.get('document_name', document.relative_path.stem)}",
                f"relative_path={document.relative_path}",
                f"section_title={section_title}",
                "",
                "Texto da seção:",
                section_text,
            ]
        )

    def _add_chunk_summaries(
        self,
        document: MarkdownDocument,
        chunks: list[MarkdownChunk],
    ) -> list[MarkdownChunk]:
        summarized_chunks: list[MarkdownChunk] = []
        for chunk in chunks:
            summary = self._summarize_chunk(document, chunk)
            summarized_chunks.append(
                MarkdownChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={**chunk.metadata, "chunk_summary": summary},
                )
            )
        return summarized_chunks

    def _summarize_chunk(self, document: MarkdownDocument, chunk: MarkdownChunk) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CHUNK_SUMMARY_PROMPT},
                {"role": "user", "content": self._build_chunk_summary_user_prompt(document, chunk)},
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError(
                f"A OpenAI retornou uma resposta vazia para o resumo do chunk {chunk.chunk_id}."
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"A resposta da LLM para resumo do chunk {chunk.chunk_id} nao veio em JSON valido."
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("chunk_summary"), str):
            raise ValueError(
                f"A resposta da LLM para resumo do chunk {chunk.chunk_id} deve conter chunk_summary string."
            )

        summary = payload["chunk_summary"].strip()
        if not summary:
            raise ValueError(
                f"A resposta da LLM para resumo do chunk {chunk.chunk_id} retornou chunk_summary vazio."
            )
        return summary

    @staticmethod
    def _build_chunk_summary_user_prompt(
        document: MarkdownDocument,
        chunk: MarkdownChunk,
    ) -> str:
        return "\n".join(
            [
                f"document_name={document.metadata.get('document_name', document.relative_path.stem)}",
                f"relative_path={document.relative_path}",
                f"chunk_index={chunk.metadata.get('chunk_index', 'n/a')}",
                "",
                "Texto do chunk:",
                chunk.text,
            ]
        )

    @staticmethod
    def _build_embedding_text(chunk: MarkdownChunk) -> str:
        document_name = str(
            chunk.metadata.get("document_name") or chunk.metadata.get("source_name") or ""
        ).strip()
        section_title = str(chunk.metadata.get("section_title", "")).strip()
        section_summary = str(chunk.metadata.get("section_summary", "")).strip()
        chunk_summary = str(chunk.metadata.get("chunk_summary", "")).strip()
        relative_path = str(chunk.metadata.get("relative_path", "")).strip()
        chunk_index = str(chunk.metadata.get("chunk_index", "")).strip()
        raw_text = chunk.text.strip()
        enriched_metadata_parts = [
            f"Documento: {document_name}" if document_name else "",
            f"Título da seção: {section_title}" if section_title else "",
            f"Resumo da seção: {section_summary}" if section_summary else "",
            f"Resumo do chunk: {chunk_summary}" if chunk_summary else "",
        ]
        raw_text_part = f"Texto do chunk:\n{raw_text}" if raw_text else ""
        enriched_metadata_text = "\n".join(part for part in enriched_metadata_parts if part)
        if enriched_metadata_text:
            return "\n".join(
                part for part in [*enriched_metadata_parts, raw_text_part] if part
            )
        fallback_parts = [
            f"relative_path={relative_path}" if relative_path else "",
            f"chunk_index={chunk_index}" if chunk_index else "",
            f"chunk_id={chunk.chunk_id}",
            raw_text_part,
        ]
        return "\n".join(part for part in fallback_parts if part)

    def _extract_keywords(self, document: MarkdownDocument) -> dict[str, str]:
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": KEYWORDS_EXTRACTION_PROMPT},
                {"role": "user", "content": document.body},
            ],
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError(
                f"A OpenAI retornou uma resposta vazia para a extracao de keywords do documento {document.relative_path}."
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"A resposta da LLM para extracao de keywords do documento {document.relative_path} nao veio em JSON valido."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"A resposta da LLM para extracao de keywords do documento {document.relative_path} deve ser um dicionario JSON."
            )

        keywords: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    f"O dicionario de keywords do documento {document.relative_path} deve conter apenas chaves e valores string."
                )
            keywords[key] = value

        return keywords

    def _save_keywords(self, document: MarkdownDocument, keywords: dict[str, str]) -> Path:
        output_path = (self.config.keywords_output_dir / document.relative_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(keywords, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("Keywords persistidas em %s", output_path)
        return output_path

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.config.embedding.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _split_front_matter(content: str) -> tuple[dict[str, object], str]:
        match = FRONT_MATTER_PATTERN.match(content)
        if not match:
            return {}, content

        raw_metadata = yaml.safe_load(match.group(1)) or {}
        if not isinstance(raw_metadata, dict):
            raise ValueError("Front matter YAML invalido: esperado objeto no topo.")

        body = content[match.end() :]
        return raw_metadata, body

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, object]) -> dict[str, object]:
        sanitized: dict[str, object] = {}
        for key, value in metadata.items():
            normalized_key = str(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[normalized_key] = value
                continue
            if isinstance(value, list):
                sanitized[normalized_key] = ", ".join(str(item) for item in value)
                continue
            sanitized[normalized_key] = WHITESPACE_PATTERN.sub(" ", str(value)).strip()
        return sanitized


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Indexa arquivos Markdown em um banco vetorial ChromaDB usando "
            "embeddings da OpenAI."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/markdown"),
        help="Diretorio contendo os arquivos markdown de entrada.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("data/chromadb"),
        help="Diretorio de persistencia do banco vetorial ChromaDB.",
    )
    parser.add_argument(
        "--keywords-output-dir",
        type=Path,
        required=True,
        help="Diretorio onde os dicionarios JSON de keywords serao gravados.",
    )
    parser.add_argument(
        "--collection",
        default="markdown_documents",
        help="Nome da colecao dentro do ChromaDB.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Modelo de embedding da OpenAI.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Quantidade de chunks enviada por requisicao de embeddings.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Tamanho maximo de cada chunk em caracteres.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Sobreposicao entre chunks consecutivos em caracteres.",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=150,
        help="Tamanho minimo para evitar fragmentos muito pequenos.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Remove a colecao antes de reindexar.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Processa somente markdowns da raiz do diretorio informado.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de log exibido durante a indexacao.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    load_dotenv()

    config = IndexingConfig(
        input_dir=args.input_dir,
        chroma_dir=args.chroma_dir,
        keywords_output_dir=args.keywords_output_dir,
        collection_name=args.collection,
        recursive=not args.non_recursive,
        reset_collection=args.reset_collection,
        chunking=ChunkingConfig(
            max_characters=args.chunk_size,
            overlap_characters=args.chunk_overlap,
            min_characters=args.min_chunk_size,
        ),
        embedding=OpenAIEmbeddingConfig(
            model=args.embedding_model,
            batch_size=args.batch_size,
        ),
    )

    summary = MarkdownVectorIndexer(config).index_documents()
    LOGGER.info(
        "Indexacao concluida: %s documentos, %s chunks, colecao=%s, chroma_dir=%s",
        summary.total_documents,
        summary.total_chunks,
        summary.collection_name,
        summary.chroma_dir,
    )
    return 0
