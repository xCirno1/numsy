from setuptools import setup, Extension

setup(
    name="fputs",
    version="1.0.0",
    description="Providing a powerful and accurate Math Solver.",
    author="xCirno1",
    author_email="xcirno6@gmail.com",
    packages=["numsy._C"],
    ext_modules=[Extension("parser", ["numsy/_C/parser.c"])]
)
