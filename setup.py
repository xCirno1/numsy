from setuptools import setup, Extension

setup(name="fputs",
          version="1.0.0",
          description="Providing a powerful and accurate Math Solver.",
          author="xCirno1",
          author_email="xcirno6@gmail.com",
          ext_modules=[Extension("fputs", ["numsy/_C/parser.c"])])
