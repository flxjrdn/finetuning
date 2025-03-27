from src import utils
from src.query_generation_for_chunks import QueryGenerator


class Evaluator:
    def __init__(self, path_chunked_docs: str):
        self.chunks = utils.load_chunks(path_chunked_docs)[:3]  # todo use all chunks
        self.query_generator = QueryGenerator(path_chunked_docs)

    def run(self):
        questions = self.query_generator.get_queries_for_eval()


if __name__ == "__main__":
    evaluator = Evaluator("chunked_documents.json")
    evaluator.run()
