import os
import sys
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from markdown_vector_indexer import (
    ChunkingConfig,
    IndexingConfig,
    MarkdownVectorIndexer,
    OpenAIEmbeddingConfig,
)

def main():

    print('Starting Markdown Vector Indexing...')

    config = IndexingConfig(
        input_dir = Path("data/markdown"),
        chroma_dir  = Path(os.getenv('CHROMA_PERSIST_DIRECTORY')),
        keywords_output_dir = Path("data/keywords"),
        collection_name = "markdown_documents",
        reset_collection = True,
        chunking = ChunkingConfig(
            max_characters = 2400,
            overlap_characters = 240,
            min_characters = 150,
        ),
        embedding = OpenAIEmbeddingConfig(
            # model = "text-embedding-3-small",
            model = os.getenv('OPENAI_EMBEDDING_MODEL'),
            batch_size = 32,
        ),
    )

    indexer = MarkdownVectorIndexer(config)
    summary = indexer.index_documents()

    print('Indexing completed. Summary:')

    print(summary)


if __name__ == "__main__":
    
    main()