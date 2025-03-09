from typing import List, Dict

import chromadb
from torch import Tensor

PATH_CHROMA_DB = "../chroma_db"
NAME_COLLECTION = "pdf_docs"


class VectorStore:
    def __init__(self):
        print(f"initializing Chroma DB at {PATH_CHROMA_DB}")
        self.chroma_client = chromadb.PersistentClient(path=PATH_CHROMA_DB)
        self.collection = self.chroma_client.get_or_create_collection(
            name=NAME_COLLECTION
        )

    def query(self, query_embedding: Tensor, n_chunks: int) -> List[str]:
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_chunks,
        )
        retrieved_texts = [result["text"] for result in results["metadatas"][0]]
        return retrieved_texts

    def empty_out_collection(self):
        if self.collection.count() == 0:
            return
        print(f"deleting collection {NAME_COLLECTION}")
        self.chroma_client.delete_collection(NAME_COLLECTION)
        print(f"creating empty collection {NAME_COLLECTION}")
        self.collection = self.chroma_client.get_or_create_collection(
            name=NAME_COLLECTION
        )

    def recreate_collection(
        self,
        chunks: List[Dict[str, str | int]],
        embedded_chunks: List[Tensor],
    ):
        self.empty_out_collection()
        for i, chunk in enumerate(chunks):
            self.collection.add(
                ids=[str(i)],
                embeddings=[embedded_chunks[i].tolist()],
                metadatas=[{"text": chunk["text"]}],
            )


if __name__ == "__main__":
    vector_store = VectorStore()
    print(f"size of collection: {vector_store.collection.count()}")
    print(f"first elements of collection: {vector_store.collection.peek()}")
