from typing import List

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from torch import Tensor

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-de"


class Embedder:
    def __init__(self):
        print(f"loading embedding model {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        self.embeddings: List[Tensor] = []

    def embed_chunks(self, chunks: List[Document]) -> None:
        self.embeddings = [
            self.embed(chunk.page_content)
            for chunk in chunks
        ]

    def embed(self, text: str) -> Tensor:
        return self.embedding_model.encode(text, convert_to_numpy=False)


if __name__ == "__main__":
    embedder = Embedder()
    texts = [
        "Welche Leistungen sind in meiner Versicherung abgedeckt?",
        "Welche Zahlungen werden mir von meinem Tarif erstattet?",
        "Wie viel Taschengeld bekomme ich?",
        "Wie heißt Du?",
    ]
    embedded_texts = [
        embedder.embed(text) for text in texts
    ]
    similarities = {
        texts[i]: [embedder.embedding_model.similarity(
            embedded_texts[i],
            embedded_texts[j],
        ) for j in range(len(texts))]
        for i in range(len(texts))
    }
    print(similarities)
