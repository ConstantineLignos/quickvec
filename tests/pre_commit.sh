#!/usr/bin/env bash
set -euxo pipefail

files=(quickvec/ tests/ setup.py)
isort -rc "${files[@]}"
black "${files[@]}"
flake8 "${files[@]}"
mypy "${files[@]}"
pytest --cov-report term-missing --cov=quickvec tests/
