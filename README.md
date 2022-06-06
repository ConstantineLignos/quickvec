# QuickVec

[![build](https://github.com/ConstantineLignos/quickvec/actions/workflows/main.yml/badge.svg)](https://github.com/ConstantineLignos/quickvec/actions/workflows/main.yml)

QuickVec is a simple package to make it easy to work with word embeddings.
QuickVec supports instantaneous loading of word embeddings after converting
them to a native SQLite format. QuickVec is designed to do exactly one thing
well: allow you to quickly load word embeddings and look up the vectors for
words.

# Installation

`pip install quickvec` (requires Python 3.6+)

# Design philosophy

QuickVec was created to support [NERPy](https://github.com/ConstantineLignos/nerpy),
a named entity recognition framework that uses word embeddings for feature
generation. NERPy originally used gensim, but the time and memory required to
load a word embedding completely into memory was a large performance
bottleneck. NERPy then turned to Magnitude, but its conversion process is quite
slow, and its installation process caused problems for NERPy users.
The NERPy developers created QuickVec based on the design of Magnitude,
but with the goal of creating a package with minimal features and dependencies.

# FAQ

* _How does QuickVec compare to [gensim](https://pypi.org/project/gensim/)'s
  `KeyedVectors` for loading word embeddings?_
  QuickVec can load word embeddings instantaneously after conversion to its
  native SQLite-based format, and does not load the whole embedding into memory,
  making it more memory efficient. However, QuickVec only supports text-format
  word embeddings files, and in general has far less functionality.
* _How does QuickVec compare to [Magnitude](https://pypi.org/project/pymagnitude/)
  for loading word embeddings?_
  Like Magnitude, QuickVec can instantly load word embeddings after conversion
  to its native SQLite-based format. QuickVec's conversion process is faster
  than Magnitude's. However, QuickVec does not support many of Magnitude's
  features, such as word similarity or generating vectors for out-of-vocabulary
  words, and QuickVec does not provide pre-converted word embeddings and only
  supports loading from text-format embeddings.
