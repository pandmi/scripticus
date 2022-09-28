from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scripticus",
    version="2.0.0",
    author="Dmitry Pankov",
    author_email="dp@pandmi.com",
    description="A package for working with the MediaMath T1 ecosystem, creating reports, dashboards, and automating campaign optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pandmi/scripticus",
    packages=find_packages(),
    install_requires=[
        "csv",
        "datetime",
        "email",
        "http.client",
        "io",
        "IPython.display",
        "ipywidgets",
        "json",
        "json",
        "urllib.request",
        "logging",
        "lxml",
        "matplotlib",
        "numpy",
        "os",
        "pandas",
        "pathlib",
        "qds_sdk",
        "re",
        "requests",
        "seaborn",
        "smtplib",
        "ssl",
        "sqlalchemy",
        "time",
        "warnings"
       ],
    python_requires='>3.1.*',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)

