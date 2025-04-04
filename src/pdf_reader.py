import json
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader

DOCUMENTS_JSON = "documents.json"


class PdfReader:
    def __init__(self, directory_pdfs: str):
        self.directory_pdfs = directory_pdfs

    def create_doc_corpus_json_file(self):
        if os.path.exists(DOCUMENTS_JSON):
            print(f"doc-corpus {DOCUMENTS_JSON} already exists")
            return
        docs = {}
        for path_pdf in self._get_path_pdfs():
            print(f"reading {os.path.basename(path_pdf)}")
            loader = PyPDFLoader(path_pdf)
            pdf_doc = loader.load()[0]
            doc_id = os.path.splitext(os.path.basename(path_pdf))[0]
            docs[doc_id] = pdf_doc.page_content
        with open(DOCUMENTS_JSON, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=4, ensure_ascii=False)
        print(f"written {len(docs)} to {DOCUMENTS_JSON}")

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


if __name__ == "__main__":
    pdf_reader = PdfReader(
        "/Users/felixjordan/Documents/code/projects/simple_scrape/pdf_downloads/"
        "signal-iduna.de"
    )
    pdf_reader.create_doc_corpus_json_file()
