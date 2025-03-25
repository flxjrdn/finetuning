from src.chunking import Chunker, CHUNKS_JSON
from src.embedding import Embedder
from src.pdf_reader import PdfReader, DOCUMENTS_JSON
from src.vector_store import VectorStore


PATH_PDF_FILES = (
    "/Users/felixjordan/Documents/code/projects/simple_scrape/pdf_downloads/"
    "signal-iduna.de"
)


class Ingestor:
    def __init__(self):
        self.pdf_reader = PdfReader(PATH_PDF_FILES)
        self.chunker = Chunker(DOCUMENTS_JSON)
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def perform_ingestion(self):
        print("started ingestion...")
        # self.pdf_reader.create_doc_corpus_json_file()
        self.chunker.create_chunks()
        self.embedder.embed_chunks(CHUNKS_JSON)
        self.vector_store.recreate_collection(
            chunks=self.chunker.chunks,
            embedded_chunks=self.embedder.embeddings,
        )


if __name__ == "__main__":
    ingestor = Ingestor()
    ingestor.perform_ingestion()
    print(f"size of collection: {ingestor.vector_store.collection.count()}")
    print(f"first elements of collection: {ingestor.vector_store.collection.peek()}")
