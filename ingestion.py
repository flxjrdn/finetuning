import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


class Chunker:
    def __init__(self, directory_pdfs: str) -> None:
        self.directory_pdfs = directory_pdfs
        self.chunks = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def create_chunks(self):
        for path_pdf in self._get_path_pdfs()[
            :3
        ]:  # TODO for testing purposes use only few files
            self._create_chunks_for_pdf(path_pdf)

    def _get_path_pdfs(self) -> List[str]:
        """
        Returns a list of full paths to all PDF files in the specified directory.

        :return: List of PDF file paths.
        """
        if not os.path.isdir(self.directory_pdfs):
            raise ValueError(f"Invalid directory: {self.directory_pdfs}")

        return [
            os.path.join(self.directory_pdfs, file)
            for file in os.listdir(self.directory_pdfs)
            if file.lower().endswith(".pdf")
        ]

    def _create_chunks_for_pdf(self, path_pdf: str):
        print(f"chunking {path_pdf}")
        loader = PyPDFLoader(path_pdf)
        pdf_doc = loader.load()
        chunks_pdf = self.text_splitter.split_documents(pdf_doc)
        self.chunks.extend(chunks_pdf)


if __name__ == "__main__":
    chunker = Chunker(
        "/Users/felixjordan/Documents/code/projects/simple_scrape/pdf_downloads/"
        "signal-iduna.de"
    )
    chunker.create_chunks()
    print(f"created {len(chunker.chunks)} chunks")
