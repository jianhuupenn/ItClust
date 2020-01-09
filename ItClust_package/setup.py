import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ItClust", 
    version="0.0.5",
    author="Jian Hu",
    author_email="jianhu@pennmedicine.upenn.edu",
    description="An Iterative Transfer learning algorithm for scRNA-seq Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jianhuupenn/ItClust",
    packages=setuptools.find_packages(),
    install_requires=["keras","pandas","numpy","scipy","scanpy","anndata","natsort","sklearn"],
    #install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
