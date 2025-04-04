import json
import os
from typing import List

from src import utils
from groq import Groq

from src.utils import get_hash_for

MODEL = "llama-3.3-70b-versatile"
TOKEN = os.environ.get("GROQ_API_TOKEN")
TEMPERATURE = 0.2
MAX_COMPLETION_TOKENS = 512


_FOLDER_INDIVIDUAL_QUERIES = "queries"


class QueryGenerator:
    def __init__(self, path_chunked_docs: str):
        if not os.path.isdir(_FOLDER_INDIVIDUAL_QUERIES):
            os.makedirs(_FOLDER_INDIVIDUAL_QUERIES, exist_ok=True)
        self.chunks = utils.load_chunks(path_chunked_docs)
        self.client = Groq(
            api_key=TOKEN,
        )
        self.queries = []

    def get_queries_for_finetuning(self) -> List[str]:
        self._get_all_queries()
        return [queries[0] for queries in self.queries]

    def get_queries_for_eval(self) -> List[str]:
        self._get_all_queries()
        return [queries[1] for queries in self.queries]

    def _get_all_queries(self):
        if os.path.exists(self._get_path_questions_json()):
            self._load_questions_from_file()
            return

        print("creating questions for evaluation and finetuning for each chunk")
        self._generate_queries_for_each_chunk()
        with open(self._get_path_questions_json(), "w", encoding="utf-8") as f:
            json.dump(self.queries, f, indent=4)

    def _get_path_questions_json(self):
        return f"queries_for_chunks_{len(self.chunks)}.json"

    def _load_questions_from_file(self):
        print(
            f"loading previously created questions from {self._get_path_questions_json()}"
        )
        with open(self._get_path_questions_json(), "r", encoding="utf-8") as f:
            self.queries = json.load(f)

    def _generate_queries_for_each_chunk(self):
        print(f"generating queries for {len(self.chunks)} chunks")
        queries_unformatted = [
            self._generate_queries_unformatted(chunk) for chunk in self.chunks
        ]
        for i in range(len(self.chunks)):
            query_unf = queries_unformatted[i]
            if ("Anfrage1: " not in query_unf) or ("Anfrage2: " not in query_unf):
                self.queries.append([])
            else:
                query1 = query_unf[len("Anfrage1: ") : query_unf.find("Anfrage2: ")]
                query2 = query_unf[query_unf.find("Anfrage2: ") + len("Anfrage2: ") :]
                query1 = query1.strip("\n")
                query2 = query2.strip("\n")
                self.queries.append([query1, query2])

    def _generate_queries_unformatted(self, text):
        if os.path.exists(self._get_path_queries_for(text)):
            with open(self._get_path_queries_for(text), "r") as f:
                return f.read()
        prompt = (
            f"Erzeuge zwei möglichst unterschiedliche Anfragen in natürlicher Sprache, die ein Nutzer stellen würde, "
            f"um den folgenden TEXT zu finden. Antworte nur mit genau den zwei Anfragen ohne weitere Informationen. "
            f"Antworte in folgendem Format:"
            f"Anfrage1: ...\n"
            f"Anfrage2: ...\n"
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
        result = chat_completion.choices[0].message.content
        with open(self._get_path_queries_for(text), "w") as f:
            f.write(result)
        return result

    @staticmethod
    def _get_path_queries_for(text: str) -> str:
        return os.path.join(_FOLDER_INDIVIDUAL_QUERIES, get_hash_for(text) + ".txt")


if __name__ == "__main__":
    qg = QueryGenerator("chunked_documents.json")
    qg.get_queries_for_finetuning()
