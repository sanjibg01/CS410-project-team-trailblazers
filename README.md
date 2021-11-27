# UIUC CS 410 - Team Trailblazers Course Project

# About

Please see `CS_410_ Project Proposal.docx` for high-level details of the project (this was a plan written at the outset of the project, so that the final result differ in some ways based on learnings along the way).

## The arXiv dataset and models

### Preprocessing

A dataset on academic journal articles was accessed from [Kaggle](https://www.kaggle.com/Cornell-University/arxiv). The dataset is contained in a single json file. Each journal article is represented by several attributes, of which our analysis uses three: Title, Abstract, and Category.

#### Initial Filtering

As an initial preprocessing step, we filtered the dataset to the categories "Computer Science - Artificial Intelligence", "Computer Science - Information Retrival", and "Computer Science - Machine Learning". This initial filtering was performed by running a bash command:

`grep arxiv-metadata-oai-snapshot.json -e "cs.AI" -e "cs.IR" -e "cs.LG" > input.json`

This initial filtering allowed us to focus on the topics of interest, and also cut down the size of the dataset considerably which was advantegous when exploring different models. Also, we created a smaller subset of the dataset (`input_small.json`) which contains roughly 1% of the documents contained in the larger `input.json`. This smaller data set was used to test code and to run initial models very quickly.

This filtered input is contained in the `arxiv/input/` directory.

#### Preprocessing the Corpus

The preprocessing code is contained in the file `data_preprocessor.py`. The following bash lines were run to perform the preprocessing:

`python data_preprocessor.py --input input/input_small.json --output preprocessed_input/preprocessed_input_small_unigrams.json`  
`python data_preprocessor.py --input input/input_small.json --output preprocessed_input/preprocessed_input_small_ngrams.json --ngrams True`

`python data_preprocessor.py --input input/input.json --output preprocessed_input/preprocessed_input_unigrams.json`  
`python data_preprocessor.py --input input/input.json --output preprocessed_input/preprocessed_input_ngrams.json --ngrams True`

The `--ngrams` argument determines if the preprocessing will produce unigram-only output or produce n-grams in the output. "n-grams" in this case are produced by the "noun chunks" functionality implemented in the Python package Spacy.

`data_preprocessor.py` performs the following pre-processing steps:
* Tokenizes
* Lemmatizes
* Removes stop words
* Lowercases
* Limits to tokens containing only alphabetic characters
* Limits to nouns (or noun chunks, determined by the `--ngrams` option)

The output is written to `arxiv/preprocessed_input/` directory.

### Modeling

Code and commentary relating to exploring candidate models and selecting a final model is contained in the Jupyter notebook file `arxiv/topic_model_search.ipynb`. Much of the code called in this notebook is contained as functions in `topic_modeling.py`.

# Repository Guide

`arxiv/input/` - Contains raw input file for the arXiv dataset

`arxiv/preprocessed_input/` - Contains preprocessed input for the arXiv dataset

`arxiv/data_preprocessor.py` - Preprocessses the arXiv dataset. See "Preprocessing the Corpus" above for details.

`arxiv/topic_modeler.py` - Produces a topic model on the preprocessed arXiv dataset. Also includes helper functions that encapsulate model exploration code; this is called from within `topic_modeling.ipynb`.

`arxiv/topic_model_search.ipynb` - Exploratory analysis and model selection code.

`arxiv/arxiv_topic_explorer.py` - Command line application that facilitates exploring the arXiv dataset through identified topics.

# How to Run

To run the arXivv data set explorer:

1) Clone the repository locally.
2) Run `python arxiv/arxiv_topic_explorer.py`

Requirements:
* Python3
* Pandas (developed with 1.3.4 but should work with older versions)

# Team Member Contributions

@mdinauta
* `arxiv/data_preprocessor.py`
* `arxiv/topic_modeler.pu`
* `arxiv/topic_model_search.ipynb`
* `arxiv/arxiv_topic_explorer.py`
* Documentation for the above (all under "The arXiv dataset and models")