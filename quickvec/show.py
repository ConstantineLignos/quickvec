import argparse
from os import PathLike
from typing import Union

from quickvec import SqliteWordEmbedding

VOCAB = "vocab"
DIM = "dim"
LENGTH = "length"
LENGTH_DIM = "length-dim"
_ACTIONS = (DIM, LENGTH, LENGTH_DIM, VOCAB)


def show(action: str, db_path: Union[str, PathLike]) -> None:
    embed = SqliteWordEmbedding.open(db_path)
    if action == DIM:
        print(embed.dim)
    elif action == LENGTH:
        print(len(embed))
    elif action == LENGTH_DIM:
        print(len(embed), embed.dim)
    elif action == VOCAB:
        for word in embed:
            print(word)
    else:
        raise ValueError(f"Unknown action {repr(action)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=_ACTIONS)
    parser.add_argument("db_path")
    args = parser.parse_args()
    show(args.action, args.db_path)


if __name__ == "__main__":
    main()
