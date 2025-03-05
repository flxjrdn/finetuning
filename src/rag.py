from typing import List

from src.embedding import Embedder
from src.generation import AnswerGenerator
from src.retriever import Retriever
from src.vector_store import VectorStore


class Answer:
    def __init__(self, text: str, context: List[str]):
        self.text = text
        self.context = context


class Rag:
    def __init__(self):
        self.retriever = Retriever(
            vector_store=VectorStore(),
            embedder=Embedder(),
        )
        self.generator = AnswerGenerator()

    def answer(self, question: str) -> Answer:
        chunks = self.retriever.get_chunks(question)
        answer = self.generator.generate(
            question=question,
            chunks=chunks,
        )
        return Answer(
            text=answer,
            context=chunks,
        )
