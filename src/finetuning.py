import json
import os
from typing import List

from groq import Groq
from sentence_transformers import SentenceTransformer, util, losses, InputExample
from torch.utils.data import DataLoader

from src import utils


MODEL = "llama-3.3-70b-versatile"
TOKEN = os.environ.get("GROQ_API_TOKEN")
TEMPERATURE = 0.2
MAX_COMPLETION_TOKENS = 512


class Finetuner:
    def __init__(self, path_chunked_docs: str, embedding_model_name: str):
        self.client = Groq(
            api_key=TOKEN,
        )
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            trust_remote_code=True,
        )
        self.chunks = utils.load_chunks(path_chunked_docs)[:3]  # todo use all chunks
        self._create_embeddings_of_chunks()

        self.queries: List[str] = []
        self.negative: List[str] = []
        self.triplets: List[tuple[str, str, str]] = []

    def finetune(self):
        print("starting to finetune embedding model...")
        self._create_triplets_for_finetuning()
        self._finetune_with_triplets()

    def _generate_query(self, text):
        prompt = (
            f"Erzeuge eine Anfrage in natürlicher Sprache, die ein Nutzer stellen würde, "
            f"um den folgenden TEXT zu finden. Antworte ausschließlich mit der Anfrage ohne weitere Informationen. "
            f"Antworte mit nur genau einer Anfrage. "
            f"Der TEXT lautet:\n{text}"
        )
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
            ],
            model=MODEL,
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        return chat_completion.choices[0].message.content

    def _generate_query_for_each_chunk(self):
        print(f"generating queries for {len(self.chunks)} chunks")
        self.queries = [self._generate_query(chunk) for chunk in self.chunks]

    def _find_least_matching_chunk_for_each_query(self):
        print(
            "finding least similar chunks to serve as negative examples in finetuning"
        )
        for i, query in enumerate(self.queries):
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
        self._generate_query_for_each_chunk()
        self._find_least_matching_chunk_for_each_query()

        for i, query in enumerate(self.queries):
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
    for i in range(len(f.chunks)):
        print(f"query: {f.queries[i]}")
        print()
        print(f"chunk: {f.chunks[i]}")
        print(f"neg.:  {f.negative[i]}")
        print()
