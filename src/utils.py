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
    chunks = [chunk["text"] for chunk in chunks_dict]
    return [chunk for chunk in chunks if chunks is not None and len(chunk) > 0]
