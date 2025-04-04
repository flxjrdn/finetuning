from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from src import utils
from src.finetuning import FINETUNED_MODEL_NAME
from src.query_generation_for_chunks import QueryGenerator
from sentence_transformers import SentenceTransformer
import faiss


class Evaluator:
    def __init__(
        self, path_chunked_docs: str, models: Dict[str, SentenceTransformer], k: int = 5
    ):
        self.k = k
        self.chunks = utils.load_chunks(path_chunked_docs)
        self.query_generator = QueryGenerator(path_chunked_docs)
        self.models = models
        self.retrieval_indices = {}

        self.faiss_index = {}
        self.hit_scores = {}
        self.mrr_scores = {}
        for model_name in self.models.keys():
            embeddings = models[model_name].encode(
                self.chunks, show_progress_bar=True, normalize_embeddings=True
            )
            dim = embeddings.shape[1]
            index_emb = faiss.IndexFlatIP(dim)
            index_emb.add(embeddings)
            self.faiss_index[model_name] = index_emb
            self.hit_scores[model_name] = []
            self.mrr_scores[model_name] = []

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
                self.mrr_scores[model_name].append(
                    self._mrr_at_k(
                        relevant_idx=i,
                        retrieved_indices=retrieved_indices,
                    )
                )
        for model_name in self.models.keys():
            print(
                f"Hit@{self.k} for model {model_name}: {np.round(np.mean(self.hit_scores[model_name]), 2)}"
            )
            print(
                f"MRR@{self.k} for model {model_name}: {np.round(np.mean(self.mrr_scores[model_name]), 2)}"
            )
        # self._plot_mrr_comparison_of_two_models()
        self._plot_mrr_violin_of_two_models()

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

    # mean reciprocal rank
    def _mrr_at_k(self, relevant_idx: int, retrieved_indices: List[int]) -> float:
        try:
            rank = list(retrieved_indices[: self.k].tolist()[0]).index(relevant_idx) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    def _plot_mrr_violin_of_two_models(self):
        model_name_a = list(self.models.keys())[0]
        model_name_b = list(self.models.keys())[1]

        # Create a tidy DataFrame for seaborn
        data = pd.DataFrame(
            {
                "MRR": self.mrr_scores[model_name_a] + self.mrr_scores[model_name_b],
                "Model": [model_name_a] * len(self.mrr_scores[model_name_a])
                + [model_name_b] * len(self.mrr_scores[model_name_b]),
            }
        )

        # Plot
        plt.figure(figsize=(6, 5))
        sns.violinplot(
            data=data, x="Model", y="MRR", inner="box", cut=0, palette="pastel"
        )
        plt.title(f"MRR@{self.k} Distribution per Model")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluator = Evaluator(
        path_chunked_docs="chunked_documents.json",
        models={
            "model_1": SentenceTransformer(
                "jinaai/jina-embeddings-v2-base-de",
                trust_remote_code=True,
            ),
            "model_2": SentenceTransformer(
                FINETUNED_MODEL_NAME,
                trust_remote_code=True,
            ),
        },
    )
    evaluator.run()
