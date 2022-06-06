import gzip
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np


class WordEmbedding(ABC):
    """A mapping between strings and their vector representations.

    While this is a Mapping-like data structure, it does not actually
    implement the Python Mapping interface to avoid having to follow the
    strict interface for key, value, and item views.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a name for the embedding."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the word representations."""

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Return the numpy dtype used to represent each dimension of the embedding."""

    @abstractmethod
    def __contains__(self, item: str) -> bool:
        """Return whether a word has a vector associated with it.

        For implementations which can produce a vector for any string,
        this should always return True."""

    @abstractmethod
    def __getitem__(self, item: str) -> np.ndarray:
        """Return the vector associated with a word."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of words in the vocabulary.

        For implementations that are capable of producing vectors for
        words not known at training time, it is recommended to return
        the size of the vocabulary used in training."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over the words in the vocabulary."""

    @abstractmethod
    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over the words in the vocabulary and their associated vectors."""

    def keys(self) -> Iterator[str]:
        """Iterate over the words in the vocabulary."""
        return iter(self)


# TODO: Add docstrings
class SqliteWordEmbedding(WordEmbedding):
    def __init__(
        self,
        conn: sqlite3.Connection,
        length: int,
        dim: int,
        dtype: np.dtype,
        name: str,
    ):
        self.conn: sqlite3.Connection = conn
        self._len: int = length
        self._dim: int = dim
        self._dtype: np.dtype = dtype
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __contains__(self, item: str) -> bool:
        result = self.conn.execute(
            "SELECT vector FROM embedding WHERE word=?", (item,)
        ).fetchone()
        return bool(result)

    def __getitem__(self, item: str) -> np.ndarray:
        result = self.conn.execute(
            "SELECT vector FROM embedding WHERE word=?", (item,)
        ).fetchone()
        if result:
            return np.frombuffer(result[0], dtype=self._dtype)
        else:
            raise KeyError(item)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[str]:
        cur = self.conn.execute("SELECT word FROM embedding ORDER BY rowid")
        while True:
            result = cur.fetchmany()
            if not result:
                break
            for item in result:
                yield item[0]

    def items(self) -> Iterator[Tuple[str, Sequence[float]]]:
        cur = self.conn.execute("SELECT * FROM embedding ORDER BY rowid")
        while True:
            result = cur.fetchmany()
            if not result:
                break
            for item in result:
                yield item[0], np.frombuffer(item[1], dtype=self._dtype)

    def close(self) -> None:
        self.conn.close()

    @classmethod
    def open(cls, db_path: Union[str, os.PathLike]) -> "SqliteWordEmbedding":
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            name, length, dim, data_type_str = conn.execute(
                "SELECT * from metadata"
            ).fetchone()
        except sqlite3.OperationalError as err:
            raise IOError(
                f"File at path {db_path} does not exist or is not a valid "
                "QuickVec-format database"
            ) from err
        dtype = np.dtype(data_type_str)
        return SqliteWordEmbedding(conn, length, dim, dtype, name)

    @classmethod
    def convert_text_format_to_db(
        cls,
        embedding_path: Union[str, os.PathLike],
        db_path: Union[str, os.PathLike],
        *,
        overwrite: bool = False,
        limit: Optional[int] = None,
        batch_size: int = 100,
        data_type: Union[Type, np.dtype] = np.float32,
        name: Optional[str] = None,
        gzipped_input: Optional[bool] = None,
    ) -> None:
        dtype = np.dtype(data_type)
        if limit is not None and limit <= 0:
            raise ValueError("Limit must be a positive number")

        if os.path.exists(db_path):
            if overwrite:
                os.remove(db_path)
            else:
                raise IOError(f"DB already exists at {db_path}")

        # Using WAL doesn't make this any faster, so no pragmas set
        # Convert path to string since connect doesn't take a PathLike in 3.6 per mypy
        try:
            conn = sqlite3.connect(str(db_path))
        except sqlite3.OperationalError as err:
            raise IOError(
                f"Cannot open path {db_path} to create output database"
            ) from err

        if gzipped_input is None:
            gzipped_input = str(embedding_path).endswith(".gz")
        file = (
            gzip.open(embedding_path, "rt", encoding="utf8")
            if gzipped_input
            else open(embedding_path, "r", encoding="utf8")
        )

        with file as embeds:
            # Get header info
            try:
                vocab_size, dim = _parse_header(next(embeds))
            except UnicodeDecodeError as err:
                raise IOError(
                    f"Embedding file {embedding_path} does not appear "
                    "to be a valid UTF-8 file."
                ) from err
            if vocab_size == 0 or dim == 0:
                raise ValueError(
                    f"Cannot load empty embedding: vocabulary {vocab_size}; "
                    f"dimensionality {dim}"
                )
            # Truncate the vocabulary size to the limit
            if limit is not None:
                vocab_size = min(vocab_size, limit)

            # Create tables
            conn.execute(
                """CREATE TABLE embedding(
                     word TEXT PRIMARY KEY NOT NULL,
                     vector BLOB NOT NULL
                   )"""
            )

            n_loaded = 0
            batch: List[Tuple[str, bytes]] = []
            for line in embeds:
                n_loaded += 1
                splits = line.rstrip(" \n").split(" ")
                word = splits[0]
                try:
                    vec = np.array(splits[1:], dtype=dtype)
                except ValueError as err:
                    # Figure out which dimension is responsible
                    for idx, val in enumerate(splits[:1]):
                        try:
                            _ = np.array(val, dtype=dtype)
                        except ValueError:
                            raise ValueError(
                                f"Could not convert dimension {idx} with value "
                                f"{repr(val)} to float on line {n_loaded + 1} "
                                f"for word {repr(word)}"
                            ) from None
                    else:
                        # We shouldn't even be able to reach here, since some dimension
                        # should have failed to convert above. Raise the original
                        # exception with a generic message.
                        # Excluded from coverage since there is no known way to reach
                        # this.
                        raise ValueError(  # pragma: no cover
                            f"Could not convert embedding on line {n_loaded + 1} "
                            f"to float for word {repr(word)}"
                        ) from err

                if len(vec) != dim:
                    # Note that the data starts on the second line, so n_loaded is one
                    # less than the line num
                    raise ValueError(
                        f"Expected dimensionality {dim} word embedding on line "
                        f"{n_loaded + 1} for word {repr(word)}, found dimensionality "
                        f"{len(vec)}"
                    )
                batch.append((word, vec.tobytes()))
                if len(batch) == batch_size:
                    cls._insert_batch(batch, conn)
                    batch = []

                if n_loaded == limit:
                    break

        if batch:
            cls._insert_batch(batch, conn)

        if n_loaded != vocab_size:
            # We disable coverage on this because it isn't reachable with any actual
            # data. It's a sanity check.
            raise ValueError(  # pragma: no cover
                f"Size of vocabulary in header ({vocab_size}) "
                f"does not match actual vocabulary size ({n_loaded})"
            )

        # Doing one big commit at the end is fastest
        conn.commit()

        # Validate length, just to be safe
        actual_length = conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0]
        if actual_length != n_loaded:
            # We disable coverage on this because it isn't reachable with any actual
            # data. It's a sanity check. You cannot hit this with duplicate words in the
            # input since they will fail the integrity check upon insertion since word
            # is constrained to be unique as it's the primary key.
            raise ValueError(  # pragma: no cover
                f"Vocabulary size of database ({actual_length})does not match "
                f"the number of words loaded from embedding ({n_loaded})."
            )

        # Create metadata table
        conn.execute(
            """CREATE TABLE metadata(
                 name TEXT PRIMARY KEY NOT NULL,
                 length INTEGER NOT NULL,
                 dim INTEGER NOT NULL,
                 dtype TEXT NOT NULL
               )"""
        )
        if name is None:
            name = os.path.basename(embedding_path)
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?)",
            (name, actual_length, dim, dtype.str),
        )
        conn.commit()

        # Clean up
        conn.close()

    @staticmethod
    def _insert_batch(
        batch: Sequence[Tuple[str, bytes]], conn: sqlite3.Connection
    ) -> None:
        try:
            conn.executemany("INSERT INTO embedding VALUES (?, ?)", batch)
        except sqlite3.IntegrityError as err:
            raise ValueError("Duplicate word in embedding file") from err


def _parse_header(line: str) -> Tuple[int, int]:
    splits = line.split()
    if len(splits) != 2:
        raise ValueError(
            "Embedding file must begin with a line containing length and dimensions"
        )
    try:
        return int(splits[0]), int(splits[1])
    except ValueError as err:
        raise ValueError(
            "Embedding length and dimensionality must be integers"
        ) from err
