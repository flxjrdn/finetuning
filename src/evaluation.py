from typing import Dict

import numpy as np

from src import utils
from src.query_generation_for_chunks import QueryGenerator
from sentence_transformers import SentenceTransformer
import faiss


class Evaluator:
    def __init__(self, path_chunked_docs: str, models: Dict[str, SentenceTransformer]):
        self.chunks = utils.load_chunks(path_chunked_docs)[:3]  # todo use all chunks
        self.query_generator = QueryGenerator(path_chunked_docs)
        self.models = models
        self.retrieval_indices = {}

        self.faiss_index = {}
        for model_name in self.models.keys():
            embeddings = models[model_name].encode(self.chunks, show_progress_bar=True, normalize_embeddings=True)
            dim = embeddings.shape[1]
            index_emb = faiss.IndexFlatIP(dim)
            index_emb.add(embeddings)
            self.faiss_index[model_name] = index_emb

    def run(self):
        for model_name in self.models.keys():
            self.get_retrieval_indices(model_name)
        # TODO evaluate and compare

    def get_retrieval_indices(self, model_name):
        questions = self.query_generator.get_queries_for_eval()
        self.retrieval_indices[model_name] = []
        for question in questions:
            self.retrieval_indices[model_name].append(
                self.retrieve_top_k(
                    model_name=model_name, question=question
                )
            )

    def retrieve_top_k(self, model_name: str, question: str, k: int = 5):
        q_embedding = self.models[model_name].encode([question], normalize_embeddings=True)
        _, indices = self.faiss_index[model_name].search(np.array(q_embedding), k)
        return indices


if __name__ == "__main__":
    evaluator = Evaluator("chunked_documents.json")
    evaluator.run()
