import json
import os
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CHUNKS_JSON = "chunked_documents.json"


class Chunker:
    def __init__(self, path_doc_corpus_json: str) -> None:
        self.path_doc_corpus_json = path_doc_corpus_json
        self.chunks: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def create_chunks(self) -> None:
        print(f"creating chunks for docs in {self.path_doc_corpus_json}")
        chunked_docs = []
        for doc_id, text in self._load_corpus().items():
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunked_docs.append({"doc_id": doc_id, "chunk_id": i, "text": chunk})
        with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
            json.dump(chunked_docs, f, indent=4, ensure_ascii=False)
        print(f"written {len(chunked_docs)} chunks to {CHUNKS_JSON}")


    def _load_corpus(self) -> Dict[str, str]:
        """Load the document corpus from a JSON file (assumed format: {'doc_id': 'text'})."""
        with open(self.path_doc_corpus_json, "r") as f:
            return json.load(f)

if __name__ == "__main__":
    c = Chunker("documents.json")
    c.create_chunks()