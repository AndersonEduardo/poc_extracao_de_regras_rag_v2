from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversionSummary:
    total_files: int
    converted_files: int
    failed_files: int


class PdfToMarkdownConverter:
    """Converte PDFs para Markdown e salva um JSON sidecar com a estrutura completa."""

    def __init__(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        *,
        recursive: bool = True,
        save_semantic_json: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.recursive = recursive
        self.save_semantic_json = save_semantic_json
        self.converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            do_formula_enrichment=False,
            do_code_enrichment=False,
        )
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )

    def iter_pdf_files(self) -> list[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Diretorio de entrada nao existe: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Caminho de entrada nao e um diretorio: {self.input_dir}"
            )

        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        return sorted(self.input_dir.glob(pattern))

    def convert_directory(self) -> ConversionSummary:
        pdf_files = self.iter_pdf_files()
        converted = 0
        failed = 0

        for pdf_path in pdf_files:
            try:
                self.convert_pdf(pdf_path)
                converted += 1
            except Exception:
                failed += 1
                LOGGER.exception("Falha ao converter %s", pdf_path)

        return ConversionSummary(
            total_files=len(pdf_files),
            converted_files=converted,
            failed_files=failed,
        )

    def convert_pdf(self, pdf_path: Path | str) -> Path:
        source = Path(pdf_path).expanduser().resolve()
        relative_path = source.relative_to(self.input_dir)
        markdown_path = (self.output_dir / relative_path).with_suffix(".md")
        semantic_json_path = markdown_path.with_suffix(".docling.json")

        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        result = self.converter.convert(source)
        if result.status != ConversionStatus.SUCCESS:
            errors = "; ".join(error.message for error in result.errors) or "erro desconhecido"
            raise RuntimeError(f"Conversao sem sucesso para {source.name}: {errors}")

        markdown_body = result.document.export_to_markdown(
            strict_text=False,
            escape_html=True,
            image_placeholder="<!-- image -->",
            page_break_placeholder="\n<!-- page-break -->\n",
            include_annotations=True,
            mark_meta=True,
            traverse_pictures=True,
            compact_tables=False,
        )

        markdown_with_metadata = self._build_markdown_document(
            source=source,
            result=result,
            markdown_body=markdown_body,
        )
        markdown_path.write_text(markdown_with_metadata, encoding="utf-8")

        if self.save_semantic_json:
            semantic_payload = {
                "source_file": str(source),
                "relative_source_file": str(relative_path),
                "status": result.status.value,
                "pages": result.input.page_count,
                "document": result.document.export_to_dict(),
            }
            semantic_json_path.write_text(
                json.dumps(semantic_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        LOGGER.info("Convertido %s -> %s", source, markdown_path)
        return markdown_path

    def _build_markdown_document(self, *, source: Path, result, markdown_body: str) -> str:
        page_count = result.input.page_count
        errors = [error.message for error in result.errors]
        front_matter = {
            "source_file": str(source),
            "source_name": source.name,
            "document_hash": result.input.document_hash,
            "page_count": page_count,
            "conversion_status": result.status.value,
            "errors": errors,
        }

        yaml_lines = ["---"]
        for key, value in front_matter.items():
            yaml_lines.extend(self._yaml_lines_for_field(key, value))
        yaml_lines.append("---")
        yaml_lines.append("")
        yaml_lines.append(markdown_body.strip())
        yaml_lines.append("")
        return "\n".join(yaml_lines)

    @staticmethod
    def _yaml_lines_for_field(key: str, value: object) -> list[str]:
        if isinstance(value, list):
            if not value:
                return [f"{key}: []"]
            lines = [f"{key}:"]
            for item in value:
                escaped = str(item).replace("\n", " ").strip()
                lines.append(f'  - "{escaped}"')
            return lines

        escaped = str(value).replace("\n", " ").strip()
        return [f'{key}: "{escaped}"']


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Converte todos os PDFs de um diretorio para Markdown e salva um JSON "
            "sidecar com a estrutura semantica preservada pelo Docling."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Diretorio que contem os arquivos PDF de entrada.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Diretorio onde os arquivos .md e .docling.json serao gravados.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Processa apenas PDFs no nivel raiz do diretorio de entrada.",
    )
    parser.add_argument(
        "--no-semantic-json",
        action="store_true",
        help="Nao salva o arquivo .docling.json com a estrutura completa.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de log exibido durante o processamento.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    converter = PdfToMarkdownConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.non_recursive,
        save_semantic_json=not args.no_semantic_json,
    )
    summary = converter.convert_directory()

    if summary.total_files == 0:
        LOGGER.warning("Nenhum arquivo PDF encontrado em %s", args.input_dir)
        return 0

    LOGGER.info(
        "Processamento concluido. total=%s convertidos=%s falhas=%s",
        summary.total_files,
        summary.converted_files,
        summary.failed_files,
    )
    return 0 if summary.failed_files == 0 else 1
