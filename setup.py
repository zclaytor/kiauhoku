import setuptools

# Load the __version__ variable without importing the package already
exec(open("kiauhoku/version.py").read())

setuptools.setup(
    name="kiauhoku",
    version=__version__,
    author="Zachary R. Claytor",
    author_email="zclaytor@hawaii.edu",
    description="Utilities for interacting and interpolating stellar evolution models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zclaytor/kiauhoku",
    license="MIT",
    python_requires='>=3',
    install_requires=[
        'numpy', 'pandas', 'scipy', 'miniutils', 'emcee', 'pyarrow', 'numba'
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
