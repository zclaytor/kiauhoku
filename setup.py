import setuptools
import codecs
import os.path


# These two functions are just to read the version number. Sourced from
# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="kiauhoku",
    version=get_version("kiauhoku/__init__.py"),
    author="Zachary R. Claytor",
    author_email="zclaytor@hawaii.edu",
    description="Utilities for interacting and interpolating stellar evolution models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zclaytor/kiauhoku",
    license="MIT",
    python_requires='>=3',
    install_requires=[
        'requests', 'tqdm', 'numpy', 'pandas', 'scipy', 'emcee', 'pyarrow', 'numba'
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
