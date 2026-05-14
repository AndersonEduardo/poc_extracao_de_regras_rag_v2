import os
import sys
from pathlib import Path

print(os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path=os.getenv('CHROMA_PERSIST_DIRECTORY'))
collection = chroma_client.get_collection(os.getenv('CHROMA_COLLECTION_NAME'))
# collection = chroma_client.get_collection('markdown_documents')

def buscar_contexto(pergunta: str, n_results: int = 3) -> list[dict]:

    embedding = openai_client.embeddings.create(
        model=os.getenv('OPENAI_EMBEDDING_MODEL'),
        input=pergunta,
        encoding_format="float",
    ).data[0].embedding

    resultados = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
    )

    itens = []
    for i in range(len(resultados["ids"][0])):
        itens.append(
            {
                "id": resultados["ids"][0][i],
                "document": resultados["documents"][0][i],
                "metadata": resultados["metadatas"][0][i],
                "distance": resultados["distances"][0][i],
            }
        )
    return itens


def main():

    hits = buscar_contexto("Quando o rastreamento deve ser anual?")

    print()

    for hit in hits:

        print(hit["metadata"]["relative_path"], hit["distance"])
        print(hit["document"][:300])

        print()
        print('----'*20)
        print()


if __name__ == "__main__":

    main()