import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kiauhoku",
    version="1.0.4",
    author="Zachary R. Claytor",
    author_email="zclaytor@hawaii.edu",
    description="utilities for interacting and interpolating stellar evolution models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zclaytor/kiauhoku",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
