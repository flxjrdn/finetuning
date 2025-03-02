from chunking import Chunker
from embedding import Embedder
from vector_store import VectorStore


class Ingestor:
    def __init__(self):
        self.chunker = Chunker(
            "/Users/felixjordan/Documents/code/projects/simple_scrape/pdf_downloads/"
            "signal-iduna.de"
        )
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def perform_ingestion(self):
        print("started ingestion...")
        chunks = self.chunker.create_chunks()
        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.recreate_collection(
            chunks=chunks,
            embedded_chunks=embeddings,
        )


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.perform_ingestion()
