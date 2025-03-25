import json
import os

from src import utils
from groq import Groq

MODEL = "llama-3.3-70b-versatile"
TOKEN = os.environ.get("GROQ_API_TOKEN")
TEMPERATURE = 0.2
MAX_COMPLETION_TOKENS = 512


class Evaluator:
    def __init__(self, path_chunked_docs: str):
        self.chunks = utils.load_chunks(path_chunked_docs)[:3]  # todo use all chunks
        self.client = Groq(
            api_key=TOKEN,
        )

    def run(self):
        self.generate_questions()

    def generate_questions(self):
        if os.path.exists(self._get_path_questions_json()):
            self._load_questions_from_file()
            return

        print("creating questions for evaluation for each chunk")
        self._generate_query_for_each_chunk()


    def _get_path_questions_json(self):
        return f"eval_questions_{len(self.chunks)}.json"

    def _load_questions_from_file(self):
        print(
            f"loading previously created questions from {self._get_path_questions_json()}"
        )
        with open(self._get_path_questions_json(), "r", encoding="utf-8") as f:
            triplet_list = json.load(f)
        self.triplets = [tuple(triplet) for triplet in triplet_list]

    def _generate_query_for_each_chunk(self):
        print(f"generating queries for {len(self.chunks)} chunks")
        self.queries = [self._generate_query(chunk) for chunk in self.chunks]

    def _generate_query(self, text):
        prompt = (
            f"Erzeuge zwei möglichst unterschiedliche Anfragen in natürlicher Sprache, die ein Nutzer stellen würde, "
            f"um den folgenden TEXT zu finden. Antworte nur mit genau den zwei Anfragen ohne weitere Informationen. "
            f"Antworte in folgendem Format:"
            f"Anfrage1: ...\n"
            f"Anfrage2: ..."
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