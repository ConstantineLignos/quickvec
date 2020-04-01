#! /usr/bin/env python

from setuptools import find_packages, setup


def setup_package() -> None:
    setup(
        name="quickvec",
        version="0.1.0-dev",
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
        # 3.6 and up, but not Python 4
        python_requires="~=3.6",
        license="MIT",
        long_description="",
        install_requires=["numpy"],
        extras_require={
            "dev": [
                "pytest",
                "pytest-cov",
                "black==19.10b0",
                "isort",
                "flake8",
                "flake8-bugbear",
                "mypy==0.770",
                "tox",
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
        ],
        project_urls={"Source": "https://github.com/ConstantineLignos/quickvec"},
    )


if __name__ == "__main__":
    setup_package()
