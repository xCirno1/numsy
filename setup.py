from setuptools import setup, Extension, find_packages

setup(
    name="numsy",
    version="0.1.2",
    description="Providing a powerful and accurate Math Solver.",
    author="xCirno1",
    author_email="xcirno6@gmail.com",
    packages=find_packages(include=["numsy", "numsy.*"]),
    ext_modules=[
        Extension(
            "numsy._C.parser",            # fully qualified import path
            ["numsy/_C/parser.c"]         # source file
        )
    ],
)
