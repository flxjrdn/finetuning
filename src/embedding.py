import json
import os
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from torch import Tensor

EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-de"


class Embedder:
    def __init__(self):
        print(f"loading embedding model {EMBEDDING_MODEL}")
        os.environ[
            "TOKENIZERS_PARALLELISM"
        ] = "false"  # disable parallelism to avoid deadlocks
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            trust_remote_code=True,
        )
        self.embeddings: List[Tensor] = []

    def embed_chunks(self, path_chunks_json: str):
        chunks_dict = self._load_chunks(path_chunks_json)
        chunks = [chunk["text"] for chunk in chunks_dict]
        chunks = [chunk for chunk in chunks if chunks is not None and len(chunk) > 0]
        self.embeddings = [self.embed(chunk) for chunk in chunks]
        print(f"created embeddings for {len(self.embeddings)} chunks")

    def embed(self, text: str) -> Tensor:
        return self.embedding_model.encode(text, convert_to_numpy=False)

    @staticmethod
    def _load_chunks(path_chunks_json) -> List[Dict[str, str]]:
        with open(path_chunks_json, "r") as f:
            return json.load(f)


if __name__ == "__main__":
    embedder = Embedder()
    texts = [
        "Welche Leistungen sind in meiner Versicherung abgedeckt?",
        "Welche Zahlungen werden mir von meinem Tarif erstattet?",
        "Wie viel Taschengeld bekomme ich?",
        "Wie heißt Du?",
    ]
    embedded_texts = [embedder.embed(text) for text in texts]
    similarities = {
        texts[i]: [
            embedder.embedding_model.similarity(
                embedded_texts[i],
                embedded_texts[j],
            )
            for j in range(len(texts))
        ]
        for i in range(len(texts))
    }
    print(similarities)
