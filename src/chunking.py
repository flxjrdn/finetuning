import json
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

CHUNKS_JSON = "chunked_documents.json"
MIN_CHUNK_CHARACTER_LENGTH = 50


class Chunker:
    def __init__(self, path_doc_corpus_json: str) -> None:
        self.path_doc_corpus_json = path_doc_corpus_json
        self.chunks: List[Dict[str, str | int]] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def create_chunks(self) -> None:
        print(f"creating chunks for docs in {self.path_doc_corpus_json}")
        for doc_id, text in self._load_corpus().items():
            chunks_for_single_doc = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks_for_single_doc):
                self.chunks.append({"doc_id": doc_id, "chunk_id": i, "text": chunk})
        self.chunks = [
            chunk for chunk in self.chunks
            if (chunk["text"] is not None) and (len(chunk["text"]) >= MIN_CHUNK_CHARACTER_LENGTH)
        ]
        with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=4, ensure_ascii=False)
        print(f"written {len(self.chunks)} chunks to {CHUNKS_JSON}")

    def _load_corpus(self) -> Dict[str, str]:
        """Load the document corpus from a JSON file (assumed format: {'doc_id': 'text'})."""
        with open(self.path_doc_corpus_json, "r") as f:
            return json.load(f)


if __name__ == "__main__":
    c = Chunker("documents.json")
    c.create_chunks()
