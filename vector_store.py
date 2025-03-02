from typing import List

import chromadb
from langchain_core.documents import Document
from torch import Tensor

PATH_CHROMA_DB = "./chroma_db"
NAME_COLLECTION = "pdf_docs"

class VectorStore:
    def __init__(self):
        print(f"initializing Chroma DB at {PATH_CHROMA_DB}")
        self.chroma_client = chromadb.PersistentClient(path=PATH_CHROMA_DB)
        self.collection = self.chroma_client.get_or_create_collection(name=NAME_COLLECTION)

    def delete_collection(self):
        if self.collection.count() == 0:
            return
        print(f"deleting collection {NAME_COLLECTION}")
        self.chroma_client.delete_collection(NAME_COLLECTION)

    def recreate_collection(
            self,
            chunks: List[Document],
            embedded_chunks: List[Tensor],
    ):
        self.delete_collection()
        for i, chunk in enumerate(chunks):
            self.collection.add(
                ids=[str(i)],
                embeddings=[embedded_chunks[i].tolist()],  # Convert numpy array to list
                metadatas=[{"text": chunk.page_content}]
    )


if __name__ == "__main__":
    vector_store = VectorStore()
    print(f"size of collection: {vector_store.collection.count()}")
    print(f"first elements of collection: {vector_store.collection.peek()}")