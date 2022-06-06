#! /usr/bin/env python

from os import path

from setuptools import find_packages, setup


def setup_package() -> None:
    root = path.abspath(path.dirname(__file__))
    with open(path.join(root, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="quickvec",
        version="0.3.0",
        packages=find_packages(include=("quickvec", "quickvec.*")),
        # Package type information
        package_data={"quickvec": ["py.typed"]},
        # Set up scripts
        entry_points={
            "console_scripts": [
                "quickvec-convert=quickvec.convert:main",
                "quickvec-show=quickvec.show:main",
            ]
        },
        # 3.6 and up
        python_requires=">=3.6",
        license="MIT",
        description="QuickVec: Fast and easy loading and querying of word vectors",
        long_description=long_description,
        install_requires=["numpy"],
        extras_require={
            "dev": [
                "pytest",
                "pytest-cov",
                "black==22.3.0",
                "isort",
                "flake8",
                "flake8-bugbear",
                "mypy==0.961",
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        url="https://github.com/ConstantineLignos/quickvec",
        long_description_content_type="text/markdown",
        author="Constantine Lignos",
        author_email="lignos@brandeis.edu",
    )


if __name__ == "__main__":
    setup_package()
