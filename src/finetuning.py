import os
from typing import List

from groq import Groq
from sentence_transformers import SentenceTransformer

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
        self.chunks = utils.load_chunks(path_chunked_docs)
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            trust_remote_code=True,
        )
        self.queries: List[str] = []

    def finetune(self):
        print("starting to finetune embedding model...")
        self._generate_queries_for_chunks()

    def _generate_query(self, text):
        prompt = (
            f"Erzeuge eine Anfrage in natürlicher Sprache, die ein Nutzer stellen würde, "
            f"um den folgenden TEXT zu finden. Antworte auschließlich mit der Anfrage ohne weitere Informationen. "
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

    def _generate_queries_for_chunks(self):
        print(f"generating queries for {len(self.chunks)} chunks")
        self.queries = [self._generate_query(chunk) for chunk in self.chunks]


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
        print()
        print()
