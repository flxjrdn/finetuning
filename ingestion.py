from chunking import Chunker
from embedding import Embedder


class Ingestor:
    def __init__(self):
        self.chunker = Chunker(
            "/Users/felixjordan/Documents/code/projects/simple_scrape/pdf_downloads/"
            "signal-iduna.de"
        )
        self.embedder = Embedder()

    def perform_ingestion(self):
        print("started ingestion...")
        self.chunker.create_chunks()
        self.embedder.embed_chunks(self.chunker.chunks)


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.perform_ingestion()
