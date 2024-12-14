from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
  

# with open('requirements.txt') as f:
#     required = f.read().splitlines()


setup(
    name="scripticus",
    version="3.0.1",
    author="Dmitry Pankov",
    author_email="dp@pandmi.com",
    description="A package for working with the MediaMath T1 ecosystem, creating reports, dashboards, and automating campaign optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pandmi/scripticus",
    # packages=find_packages(),
    # install_requires=required,
    # python_requires > '3.1.*',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],)

