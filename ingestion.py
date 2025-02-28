from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_chunks(path_pdf: str) -> List[Document]:
    loader = PyPDFLoader(path_pdf)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    chunks = create_chunks(
        "/Users/felixjordan/Documents/code/projects/simple_scrape/pdf_downloads/"
        "signal-iduna.de/bedingungen-tierhalterhaftpflicht.pdf"
    )
    print(f"created {len(chunks)} chunks")
