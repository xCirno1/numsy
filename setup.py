from setuptools import setup, Extension, find_packages

setup(
    name="numsy",
    version="0.1.1-alpha.1",
    description="Providing a powerful and accurate Math Solver.",
    author="xCirno1",
    author_email="xcirno6@gmail.com",
    packages=find_packages(include=["numsy", "numsy.*"]),
    ext_modules=[
        Extension(
            "numsy._C.parser",
            ["numsy/_C/parser.c"]
        )
    ],
)
