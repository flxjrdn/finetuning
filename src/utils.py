import hashlib
import json
from typing import List


def load_chunks(path_chunks_json) -> List[str]:
    """
    Loads chunks from json file and returns them as list of strings.
    :param path_chunks_json: path to json file with chunks
    :return: list of all chunks
    """
    print(f"loading chunks from {path_chunks_json}...")
    with open(path_chunks_json, "r") as f:
        chunks_dict = json.load(f)
    return [chunk["text"] for chunk in chunks_dict]


def get_hash_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
