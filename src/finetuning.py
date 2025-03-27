import json
import os
from typing import List

from sentence_transformers import SentenceTransformer, util, losses, InputExample
from torch.utils.data import DataLoader

from src import utils
from src.query_generation_for_chunks import QueryGenerator


class Finetuner:
    def __init__(self, path_chunked_docs: str, embedding_model_name: str):
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            trust_remote_code=True,
        )
        self.chunks = utils.load_chunks(path_chunked_docs)[:3]  # todo use all chunks
        self._create_embeddings_of_chunks()
        self.query_generator = QueryGenerator(path_chunked_docs)

        self.negative: List[str] = []
        self.triplets: List[tuple[str, str, str]] = []

    def finetune(self):
        print("starting to finetune embedding model...")
        self._create_triplets_for_finetuning()
        self._finetune_with_triplets()

    def _find_least_matching_chunk_for_each_query(self, queries: List[str]):
        print(
            "finding least similar chunks to serve as negative examples in finetuning"
        )
        for i, query in enumerate(queries):
            # Compute similarity scores
            scores = util.pytorch_cos_sim(self.embeddings[i], self.embeddings)[0]

            # Pick least similar chunk as negative
            negative_idx = scores.argsort()[0].item()
            self.negative.append(self.chunks[negative_idx])

    def _create_triplets_for_finetuning(self):
        if os.path.exists(self._get_path_triplets_json()):
            self._load_triplets_from_file()
            return

        print("creating triplets of query, positive and negative examples")
        queries = self.query_generator.get_queries_for_finetuning()
        self._find_least_matching_chunk_for_each_query(queries)

        for i, query in enumerate(queries):
            anchor = query
            positive = self.chunks[i]
            negative = self.negative[i]
            self.triplets.append((anchor, positive, negative))

        with open(self._get_path_triplets_json(), "w", encoding="utf-8") as f:
            json.dump(self.triplets, f, indent=4)

    def _create_embeddings_of_chunks(self):
        print(f"creating embeddings of {len(self.chunks)} chunks")
        self.embeddings = self.embedding_model.encode(
            self.chunks, convert_to_tensor=True
        )

    def _finetune_with_triplets(self):
        train_data = [InputExample(texts=[a, p, n]) for a, p, n in self.triplets]
        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

        train_loss = losses.TripletLoss(self.embedding_model)
        self.embedding_model.fit(
            train_objectives=[(train_dataloader, train_loss)], epochs=3
        )

    def _get_path_triplets_json(self):
        # use number of chunks because this is known before triplets are created
        return f"query_triplets_{len(self.chunks)}.json"

    def _load_triplets_from_file(self):
        print(
            f"loading previously created triplets from {self._get_path_triplets_json()}"
        )
        with open(self._get_path_triplets_json(), "r", encoding="utf-8") as f:
            triplet_list = json.load(f)
        self.triplets = [tuple(triplet) for triplet in triplet_list]


if __name__ == "__main__":
    f = Finetuner(
        path_chunked_docs="chunked_documents.json",
        embedding_model_name="jinaai/jina-embeddings-v2-base-de",
    )
    f.finetune()
