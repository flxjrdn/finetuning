from typing import Dict, List

import numpy as np

from src import utils
from src.query_generation_for_chunks import QueryGenerator
from sentence_transformers import SentenceTransformer
import faiss


class Evaluator:
    def __init__(
        self, path_chunked_docs: str, models: Dict[str, SentenceTransformer], k: int = 5
    ):
        self.k = k
        self.chunks = utils.load_chunks(path_chunked_docs)[:3]  # todo use all chunks
        self.query_generator = QueryGenerator(path_chunked_docs)
        self.models = models
        self.retrieval_indices = {}

        self.faiss_index = {}
        self.hit_scores = {}
        for model_name in self.models.keys():
            embeddings = models[model_name].encode(
                self.chunks, show_progress_bar=True, normalize_embeddings=True
            )
            dim = embeddings.shape[1]
            index_emb = faiss.IndexFlatIP(dim)
            index_emb.add(embeddings)
            self.faiss_index[model_name] = index_emb
            self.hit_scores[model_name] = []

    def run(self):
        for model_name in self.models.keys():
            self.get_retrieval_indices(model_name)
            for i, retrieved_indices in enumerate(self.retrieval_indices[model_name]):
                # relevant index is i, because the i-th query was generated for the i-th chunk
                self.hit_scores[model_name].append(
                    self._hit_at_k(
                        relevant_idx=i,
                        retrieved_indices=retrieved_indices,
                    )
                )
        print(self.hit_scores)

    def get_retrieval_indices(self, model_name):
        questions = self.query_generator.get_queries_for_eval()
        self.retrieval_indices[model_name] = []
        for question in questions:
            self.retrieval_indices[model_name].append(
                self.retrieve_top_k(model_name=model_name, question=question)
            )

    def retrieve_top_k(self, model_name: str, question: str):
        q_embedding = self.models[model_name].encode(
            [question], normalize_embeddings=True
        )
        _, indices = self.faiss_index[model_name].search(np.array(q_embedding), self.k)
        return indices

    def _hit_at_k(self, relevant_idx: int, retrieved_indices: List[int]) -> int:
        return int(relevant_idx in retrieved_indices[: self.k])


if __name__ == "__main__":
    evaluator = Evaluator(
        path_chunked_docs="chunked_documents.json",
        models={
            "model_1": SentenceTransformer(
                "jinaai/jina-embeddings-v2-base-de",
                trust_remote_code=True,
            ),
            "model_2": SentenceTransformer(
                "jinaai/jina-embeddings-v2-base-de",
                trust_remote_code=True,
            ),
        },
    )
    evaluator.run()
