#! /usr/bin/env python

import argparse

import quickvec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings_path")
    parser.add_argument("db_path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gzipped", action="store_true", default=None)
    args = parser.parse_args()

    quickvec.SqliteWordEmbedding.convert_text_format_to_db(
        args.embeddings_path,
        args.db_path,
        limit=args.limit,
        name=args.name,
        overwrite=args.overwrite,
        gzipped_input=args.gzipped,
    )


if __name__ == "__main__":
    main()
