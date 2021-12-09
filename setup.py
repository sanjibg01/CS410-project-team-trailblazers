import setuptools

setuptools.setup(
    name="search",
    version="0.0.1",
    author="Paul Zuradzki",
    description="TF-IDF search for Coursera and Arxiv papers",
    packages=setuptools.find_packages(),
    install_requires=["click", "pandas", "tabulate", "scikit-learn"],
)