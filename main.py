#!/usr/bin/env python3

import os
import re
import json
from pathlib import Path

from pdfminer.high_level import extract_text

import settings as s
from src.database import fill_db, TextSource


# logging.basicConfig(level=logging.INFO  # TODO  init logger

DOCS_DIR = "../" + s.settings.PATH_DATA + "text_sources/"


# connect to database and insert automatically all pdf files stored in data folder
for filename in os.listdir(DOCS_DIR):
    if filename.endswith(".pdf"):
        print(f"fetching: {filename}")

        file_path = os.path.join(DOCS_DIR, filename)
        text = extract_text(file_path)
        filename = Path(filename).stem
        authors, title = authors, title = (
            re.compile(r"(.+?)[0-9]{4}(.*)?").search(filename).groups()
        )

        entry = {
            "authors": authors.strip(),
            "title": title.strip(),
            "source": "dummy source",
            "content": text,
            "metadata": {
                "tags": ["ahr_valley", "dummy_publication_type"],
                "published_date": re.findall(r"[0-9]{4}", filename)[0],
            },
        }
    fill_db(TextSource(**entry))




## TODO init main function
def main():
    pass  # placeholder


if __name__ == "__main__":
    main()
