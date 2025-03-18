import os

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

    def finetune(self):
        queries = [self.generate_query(chunk) for chunk in self.chunks]

    def generate_query(self, text):
        prompt = f"Generate a natural language search query a user might use to find this text:\n{text}\n\nQuery:"
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
