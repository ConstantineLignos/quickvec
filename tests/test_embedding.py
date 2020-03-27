import os
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from quickvec import SqliteWordEmbedding
from quickvec.embedding import _parse_header


def _data_path(filename: str) -> str:
    return os.path.join("tests", "test_data", filename)


def test_load_basic() -> None:
    for filename in ("word_vectors.vec", "word_vectors.vec.gz"):
        with TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "tmp.sqlite")
            # Batch size intentionally not aligned with vocabulary size
            embed = SqliteWordEmbedding.from_text_format(
                _data_path(filename), db_path, batch_size=2
            )

            # Test file is in float32
            expected_dtype = np.dtype(np.float32)
            expected_vocab = ["the", "Wikipedia", "article", "\t"]
            expected_vecs = {
                "the": np.array([0.0129, 0.0026, 0.0098], dtype=expected_dtype),
                "Wikipedia": np.array([0.0007, -0.0205, 0.0107], dtype=expected_dtype),
                "article": np.array([0.0050, -0.0114, 0.0150], dtype=expected_dtype),
                "\t": np.array([0.0001, 0.0002, 0.0003], dtype=expected_dtype),
            }

            assert len(embed) == len(expected_vocab)
            assert embed.dim == len(expected_vecs[expected_vocab[0]])
            # Iteration maintains the original text file order
            assert list(embed) == expected_vocab
            assert list(embed.keys()) == expected_vocab
            assert embed.dtype == expected_dtype
            assert embed.name == filename

            for word, vec in expected_vecs.items():
                assert np.array_equal(embed[word], vec)

            for word, vec in embed.items():
                assert np.array_equal(vec, expected_vecs[word])

            # Check order of items matches original order
            assert [item[0] for item in embed.items()] == expected_vocab

            for word in expected_vocab:
                assert word in embed
            assert "jdlkjalkas" not in embed
            with pytest.raises(KeyError):
                embed["asdaskl"]
            embed.close()


def test_limit() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with closing(
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/word_vectors.vec", db_path, limit=2
            )
        ) as embeds:
            assert len(embeds) == 2

    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with closing(
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/word_vectors.vec", db_path, limit=1
            )
        ) as embeds:
            assert len(embeds) == 1

    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(ValueError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/word_vectors.vec", db_path, limit=0
            )


def test_name() -> None:
    with TemporaryDirectory() as tmp_dir:
        name = "test_name"
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with closing(
            SqliteWordEmbedding.from_text_format(
                _data_path("word_vectors.vec"), db_path, name=name
            )
        ) as embeds:
            assert embeds.name == name


def test_gzipped() -> None:
    # We only need to test the optional argument with True/False as functionality with
    # None is tested in test_load_basic. We're just checking that things don't crash.
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        SqliteWordEmbedding.from_text_format(
            _data_path("word_vectors.vec"), db_path, gzipped_input=False
        )
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        SqliteWordEmbedding.from_text_format(
            _data_path("word_vectors.vec.gz"), db_path, gzipped_input=True
        )

    # The gzip file is not valid UTF-8, so we get IOError if we force it to be not
    # gzipped.
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(IOError):
            SqliteWordEmbedding.from_text_format(
                _data_path("word_vectors.vec.gz"), db_path, gzipped_input=False
            )


def test_overwrite() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        SqliteWordEmbedding.from_text_format(
            "tests/test_data/word_vectors.vec", db_path
        )
        # Cannot overwrite without specifying it
        with pytest.raises(IOError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/word_vectors.vec", db_path
            )
        # Overwriting works
        with closing(
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/word_vectors.vec", db_path, overwrite=True
            )
        ):
            pass


def test_load_bad_dims() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(ValueError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/bad_dims.badvec", db_path
            )


def test_load_bad_vocab_size() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(ValueError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/bad_vocab_size1.badvec", db_path
            )

    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(ValueError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/bad_vocab_size2.badvec", db_path
            )


def test_load_empty_vecs() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(ValueError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/empty.badvec", db_path
            )


def test_load_duplicate_words() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(ValueError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/duplicate_word.badvec", db_path
            )


def test_bad_float() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(ValueError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/bad_float.badvec", db_path
            )


def test_bad_output_path() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "nonexistent", "tmp.sqlite")
        with pytest.raises(IOError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/word_vectors.sqlite", db_path
            )


def test_bad_input_path() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with pytest.raises(IOError):
            SqliteWordEmbedding.from_text_format(
                "tests/test_data/nonexistent.sqlite", db_path
            )


def test_load_empty_db() -> None:
    with pytest.raises(IOError):
        with closing(SqliteWordEmbedding.from_db("tests/test_data/empty_db.sqlite")):
            pass


def test_path_types() -> None:
    with TemporaryDirectory() as tmp_dir:
        vec_path_str = "tests/test_data/word_vectors.vec"
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with closing(SqliteWordEmbedding.from_text_format(vec_path_str, db_path)):
            pass

    with TemporaryDirectory() as tmp_dir:
        vec_path_path = Path(vec_path_str)
        db_path = os.path.join(tmp_dir, "tmp.sqlite")
        with closing(SqliteWordEmbedding.from_text_format(vec_path_path, db_path)):
            pass

    db_path_str = "tests/test_data/word_vectors.sqlite"
    with closing(SqliteWordEmbedding.from_db(db_path_str)):
        pass

    db_path_path = Path(db_path_str)
    with closing(SqliteWordEmbedding.from_db(db_path_path)):
        pass


def test_parse_dims() -> None:
    assert _parse_header("123 456\n") == (123, 456)
    with pytest.raises(ValueError):
        _parse_header("123 5.0\n")
    with pytest.raises(ValueError):
        _parse_header("123.0 456\n")
    with pytest.raises(ValueError):
        _parse_header("3 3 3\n")
