TODO add description of preprocessing options
TODO final topic models


# UIUC CS 410 (Text Information Systems) - Team Trailblazers Course Project

# About

Please see `CS_410_ Project Proposal` for high-level details of the project (this was a plan written at the outset of the project, so that the final result differ in some ways based on learnings along the way).

## The arXiv dataset

A dataset on academic journal articles was accessed from [Kaggle](https://www.kaggle.com/Cornell-University/arxiv). The dataset is contained in a single json file. Each journal article is represented by several attributes; our analysis used a subset of these: Title, Abstract, and Category.

As an initial preprocessing step, we filtered the dataset to the categories "Computer Science - Artificial Intelligence", "Computer Science - Information Retrival", and "Computer Science - Machine Learning". This filtering was performed by running a simple bash command:

`grep arxiv-metadata-oai-snapshot.json -e "cs.AI" -e "cs.IR" -e "cs.LG" > input.json`

Filtering by category allowed us to focus the dataset down to the topics of interest, and also cut down the size of the dataset considerably which was advantegous when exploring models etc.

The preprocessing pipeline is contained in the file `data_preprocessor.py`

# Repository Guide

`arxiv/input/` - contains raw input file for the arxiv dataset

`arxiv/preprocessed_input/`

`arxiv/data_preprocessor.py` Preprocesss the arxiv dataset. See "The arXiv dataset" for details.

`arxiv/topic_modeler.py` Produce a topic model on the preprocessed arXiv dataset. Also includes helper functions that encapsulate model exploration code; this is called from within `topic_modeling.ipynb`.

`arxiv/topic_modeling.ipynb` Exploratory analysis and model selection code.

# How to Run

TBU