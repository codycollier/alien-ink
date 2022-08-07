from setuptools import setup, find_packages
exec(open('alien_ink/version.py').read())

setup(
  name = 'alien-ink',
  packages = find_packages(exclude=[]),
  version = __version__,
  license='MIT',
  description = 'Alien Ink - a personal machine learning toolkit',
  author = 'Cody Collier',
  author_email = 'cody@telnet.org',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/codycollier/alien-ink',
  keywords = [
    'artificial intelligence',
    'machine learning',
  ],
  install_requires=[
    'kaggle',
    'numpy',
    'packaging',
    'pandas',
    'sklearn',
    'torch>=1.6',
    'transformers',
    'tqdm',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
