from setuptools import setup

setup(
  name='scripticus',
  version='1.2.2',
  author='Dmitry Pankov',
  author_email='dpankov@mediamath.com',
  packages=['scripticus'],
  scripts=['scripticus/beautifulization','scripticus/looker_api','scripticus/mailicus','scripticus/t1_api'],
  url='https://github.com/pandmi/scripticus/',
  license='LICENSE.txt',
  description='An absolutly awesome package that does everything!',
  long_description=open('README.txt').read(),
  install_requires=[
  "pandas","numpy","seaborn","requests","os","ipywidgets",
  "IPython.display","json",
  "http","seaborn","matplotlib","csv","time","qds_sdk",
  "sqlalchemy","datetime","pathlib","lxml","datetime","warnings",
  "logging","requests","smtplib","ssl","os","email",
  ],
)




