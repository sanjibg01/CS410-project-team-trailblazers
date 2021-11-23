# UIUC CS 410 (Text Information Systems) - Team Trailblazers Course Project

# About

Please see `CS_410_ Project Proposal` for high-level details of the project (this was a plan written at the outset of the project, so that the final result differ in some ways based on learnings along the way).

## The arXiv dataset and models

### Preprocessing

A dataset on academic journal articles was accessed from [Kaggle](https://www.kaggle.com/Cornell-University/arxiv). The dataset is contained in a single json file. Each journal article is represented by several attributes. Our analysis used a subset of these attributes: Title, Abstract, and Category.

As an initial preprocessing step, we filtered the dataset to the categories "Computer Science - Artificial Intelligence", "Computer Science - Information Retrival", and "Computer Science - Machine Learning". This initial filtering was performed by running a bash command:

`grep arxiv-metadata-oai-snapshot.json -e "cs.AI" -e "cs.IR" -e "cs.LG" > input.json`

This initial filtering allowed us to focus on the topics of interest, and also cut down the size of the dataset considerably which was advantegous when exploring different models. Also, we created a smaller subset of the dataset (`input_small.json`) which contains approx. 1% of the data in the larger `input.json`. This smaller data set was used to quickly run initial models.

The preprocessing pipeline is contained in the file `data_preprocessor.py`. The following bash lines were run to perform the preprocessing:

`python data_preprocessor.py --input input/input_small.json --output preprocessed_input/preprocessed_input_small_unigrams.json`
`python data_preprocessor.py --input input/input_small.json --output preprocessed_input/preprocessed_input_small_ngrams.json --ngrams True`

`python data_preprocessor.py --input input/input.json --output preprocessed_input/preprocessed_input_unigrams.json`
`python data_preprocessor.py --input input/input.json --output preprocessed_input/preprocessed_input_ngrams.json --ngrams True`

The `--ngrams` argument determines if the pre-processing will produce unigram-only output or produce n-grams in the output. "n-grams" in this case are produced by the "noun chunks" functionality in the Python package Spacy.

`data_preprocessor.py` performs the following pre-processing steps:
* Tokenizes
* Lemmatizes
* Removes stop words
* Lowercases
* Limits to tokens containing only alphabetic characters
* Limits to nouns (or noun chunks, determined by the `--ngrams` option)

The output is written to the `preprocessed_input` directory. 

### Modeling

Code and commentary relating to exploring candidate models and selecting a final model is contained in the Jupyter notebook file `arxiv/topic_model_search.ipynb`. Much of the code called in this notebook is contained as functions in `topic_modeling.py`.

# Repository Guide

`arxiv/input/` - Contains raw input file for the arXiv dataset

`arxiv/preprocessed_input/` - Contains preprocessed input for the arXiv dataset (i.e raw input run through `data_preprocessor.py`)

`arxiv/data_preprocessor.py` - Preprocessses the arXiv dataset. See "Preprocessing" above for details.

`arxiv/topic_modeler.py` - Produces a topic model on the preprocessed arXiv dataset. Also includes helper functions that encapsulate model exploration code; this is called from within `topic_modeling.ipynb`.

`arxiv/topic_model_search.ipynb` - Exploratory analysis and model selection code.

# How to Run

TBU

# Team Member Constributions

@mdinauta
* `arxiv/data_preprocessor.py`
* `arxiv/topic_modeler.pu`
* `arxiv/topic_model_search.ipynb`
* Documentation for the above (all under "The arXiv dataset and models")