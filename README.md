# UIUC CS 410 - Team Trailblazers Course Project
Write-up/documentation on the final project.

# About
Please see `CS_410_ Project Proposal.docx` for high-level details of the project (this was a plan written at the outset of the project, so that the final result differ in some ways based on learnings along the way).

## The arXiv dataset and models

For a video presentation that covers this part of the project, please see https://drive.google.com/file/d/1HwzMQZjem2aQkvqJqjUOAcZjn0SjSZv1/view?usp=sharing.

### Preprocessing

A dataset on academic journal articles was accessed from [Kaggle](https://www.kaggle.com/Cornell-University/arxiv). The dataset is contained in a single json file. Each journal article is represented by several attributes of which our analysis uses three: Title, Abstract, and Category.

#### Initial Filtering

As an initial preprocessing step, we filtered the dataset to the categories "Computer Science - Artificial Intelligence", "Computer Science - Information Retrival", and "Computer Science - Machine Learning". This initial filtering was performed by running a bash command:

`grep arxiv-metadata-oai-snapshot.json -e "cs.AI" -e "cs.IR" -e "cs.LG" > input.json`

This initial filtering allowed us to focus on the topics of interest, and also cut down the size of the dataset considerably which was advantegous when exploring different models. Also, we created a smaller subset of the dataset (`input_small.json`) which contains roughly 1% of the documents contained in the larger `input.json`. This smaller data set was used for testing code and for running initial models very quickly.

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
* Limits to nouns (or "noun chunks", as determined by the `--ngrams` option)

The output is written to `arxiv/preprocessed_input/` directory.

### Modeling

Code and commentary relating to exploring candidate models and selecting a final model is contained in the Jupyter notebook file `arxiv/topic_model_search.ipynb`. To make the notebook more readable, much of the code called in this notebook is contained as functions in `topic_modeling.py` which are then imported in the notebook.


## Mining topics from lecture transcripts

For a video presentation that covers this part of the project, please see ____.
Detailed documentation is in the form of comments in this python file mine_lecture_topics.py. It can run standalone, requiring as inputs the transcripts in json format. 


### Data Source
Transcript data from the CS410 course (comprising of two MOOCs: Text Retrieval; Text Mining and Analytics) are obtained using the python package coursera-dl. We use this course as an example to show enhanced intelligent learning. However, the topic mining is general enough that can be applied to any course lecture transcripts. 100 documents of lecture video transcripts were available and collected. 

### Text Pre-processing
The library metapy is used to create a token stream that applies these text pre-processing steps to the lecture transcripts:
1.	Lower case
2.	Character length filter 
3.	Lemmatize 
4.	Filter out common stop words or words in transcript data that are not necessary i.e. [MUSIC], [INAUDIBLE], [SOUND]
5.	Tokenize into unigrams
These pre-processing steps are used as they produced the most robust word units to mine the text data (since the transcripts are verbal, not written language, applying n-grams produced many word combinations that would not produce as much significant meaning as from written documents i.e. research papers).

### Topic Mining
The LDA model from the genism library is used to mine multiple topics from the processed text data in the form of Bag of Words with TF-IDF weighting to consider rare terms more. A grid search through a range of hyperparameters is conducted to find the most optimal model that produces high topic coherence and topic distribution across all 100 documents. Since the number of documents (100) is small, the LDA tends to be unstable producing topics with low average topic coherence (range 0.3-0.5) with imbalanced number of documents within each topic. To address this in future work, we would explore supplementing transcript data from other text mining courses to mine for more robust topics. We explore whether the topics found can help retrieve similar documents to a search query. With the most optimal LDA model found, an output dataframe is generated containing the dominant topic for each document, topic coverage, and most salient keywords of the dominant topic. These features are used in the text retrieval process to help retrieve lectures relevant to the query and similar to other lectures in the same topic. 

# Repository Guide

## arXiv topic modeling

`arxiv/input/` - Contains raw input file for the arXiv dataset

`arxiv/preprocessed_input/` - Contains preprocessed input for the arXiv dataset

`arxiv/data_preprocessor.py` - Preprocessses the arXiv dataset. See "Preprocessing the Corpus" above for details.

`arxiv/topic_modeler.py` - Produces a topic model on the preprocessed arXiv dataset. Also includes helper functions that encapsulate model exploration code; this is called from within `topic_modeling.ipynb`.

`arxiv/topic_model_search.ipynb` - Exploratory analysis and model selection code.

`arxiv/arxiv_topic_explorer.py` - Command line application that facilitates exploring the arXiv dataset through identified topics.

## Lecture transcript topic modeling

`lecture_transcript_mining/transcripts_text-retrieval_txt.json` – Contains lecture transcripts downloaded using coursera-dl

`lecture_transcript_mining/transcripts_text-mining_txt.json` – Contains lecture transcripts downloaded using coursera-dl

`lecture_transcript_mining/mine_lecture_topics.py` – Conducts text pre-processing and topic mining on lecture transcripts

`lecture_transcript_mining/doc_topic_summary.csv` – Output topic distribution of the mine_lecture_topics.py

`lecture_transcript_mining/topic mining exploration.ipynb` - Exploration of the topic mining process

# How to Run

To run the arXiv data set explorer:

1) Clone the repository locally.
2) Run `python arxiv/arxiv_topic_explorer.py`

Requirements:
* Python3
* Pandas (developed with 1.3.4 but should work with older versions as well)

To run the lecture topic mining:

1) Clone the repository locally.
2) Run `python lecture_transcript_mining/mine_lecture_topics.py`

Requirements:
* Python3


# Team Member Contributions

@mdinauta (Matt DiNauta)
* `arxiv/data_preprocessor.py`
* `arxiv/topic_modeler.py`
* `arxiv/topic_model_search.ipynb`
* `arxiv/arxiv_topic_explorer.py`
* Documentation for the above (i.e. all documentation under heading "The arXiv dataset and models")

@boryehn (Bo-Ryehn Chung)
* `lecture_transcript_mining/mine_lecture_topics.py` (covers text preprocessing, model search, and topic mining in one script)
* `lecture_transcript_mining/topic mining exploration.ipynb`
* Documentation under heading "Mining topics from lecture transcripts"
