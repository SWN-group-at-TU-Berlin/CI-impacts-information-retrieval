#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Connect and inset to vector DB"""

__author__ = "Anna Buch, TU Berlin"
__email__ = "anna.buch@tu-berlin.de"


from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
import json
from typing import Optional, Dict, Any

import psycopg2 as pg


# logger = s.init_logger("__database__") # TODO init logger in settings


class TextSource(BaseModel):
    """
    Data model to ensure a fixed structure for the text source entries in the database
    """

    id: UUID = Field(default_factory=uuid4)  #  unique entry id to prevent overwriting
    title: str
    source: str
    content: str
    authors: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # make model immutable
    model_config = ConfigDict(frozen=True)


def connect_db():
    conn = pg.connect(
        host="localhost",  # TODO configure as env variables e.g os.environ.get('PG_HOST')
        user="postgres",
        dbname="postgres",
        port="5432",
        password="postgres",
    )
    print(conn)
    return conn


def fill_db(
    entry: TextSource, schema_name: str = "public", table_name: str = "text_sources"
):
    conn = connect_db()
    curs = conn.cursor()

    curs.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema_name}.{table_name}(
            id SERIAL PRIMARY KEY,
            title TEXT,
            authors TEXT,
            source TEXT,
            content TEXT,
            metadata JSONB
        );
        """
    )

    curs.execute(
        f"""
        INSERT INTO {schema_name}.{table_name}(title, authors, source, content, metadata)
        VALUES
            ('{entry.authors}',
            '{entry.title}',
            '{entry.source}',
            '{entry.content}',
            '{json.dumps(entry.metadata)}'
        );
        """
    )
    conn.commit()

    # logger.info(f"Inserted entry '{entry.title}' into database.")

    curs.close()
    conn.close()
