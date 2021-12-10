# UIUC CS 410 - Team Trailblazers Course Project
Write-up/documentation on the final project, with links to presentation within each main section.

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

## TF-IDF and Cosine Similarity Search for Querying Lectures and Arxiv Papers
### Description of `search/search.py`
* Structure
    * `SearchEngine` class with methods for loading Arxiv papers, Coursera lectures, and running queries
    * `TfidfCosineSearch` class that scores documents against a query. 
        * This class can be applied generally to other contexts. 
        * By default, will print a tabular display of the top 10 documents matching the query sorted in descending order 
    * Command line interface functions (decorated with `@click`) that specify how a user can use the search tool from command line
* `scikit-learn`
    * `scikit-learn` is used to implement Term-Frequency Inverted Document Frequency weights.
    * `scikit-learn` is also used for to rank documents to a query via cosine similarity. 
        * While we can hand-roll our own algorithms, `scikit-learn` has useful optimizations for larger datasets (use of numpy, sparse arrays) that let us focus on delivering the app. `scikit-learn` uses plus-one smoothing to avoid zero probability weights, as we discuss in this course.
* We use `importlib.resources` to deliver sample data via pip.
* We use the `click` library to help specify our command-line interface with out-of-the-box usage documentation.
* The `./search/data` directory contains our Coursera lecture transcripts and a sample of Arxiv papers for demonstration.
    * The Arxiv data is available as a 3GB download from Kaggle or through a Google API. This package includes a small 10,000 document sample.
    * The Coursera transcripts were scraped using `coursera-dl` (external Python CLI app).
    * The Coursera transcripts were further processed and merged into one collection, since `coursera-dl` splits the transcripts into a directory for each week.
        *  Processing work: https://github.com/sanjibg01/CS410-project-team-trailblazers/blob/paulzuradzki/scripts/fetch_transcripts.py

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


## Search Command Line Interface
The `./search` directory contains a Python package that is installable via pip.
* `./search/search.py` is the main module that implements TF-IDF and cosine similarity search. This module also specifies the command line interface.
* `./search/data` contains Coursera transcripts (obtained via courersa-dl scraper package) and a small sample of Arxiv data (via Kaggle) for demonstration purposes. These files get installed with pip.

# How to Run

### To run the arXiv data set explorer:

1) Clone the repository locally.
2) Run `python arxiv/arxiv_topic_explorer.py`

Requirements:
* Python3
* Pandas (developed with 1.3.4 but should work with older versions as well)

### To run the lecture topic mining:

1) Clone the repository locally.
2) Run `python lecture_transcript_mining/mine_lecture_topics.py`

Requirements:
* Python3

### To run the Search CLI:

Installation with pip
```bash
# create a virtual environment
$ python -m venv venv

# Activate the virtual environment with the command
$ source venv/bin/activate       # Mac/Linux
$ source ./venv/Scripts/activate # Windows

# upgrade pip
(venv) $ python -m pip install --upgrade pip

# install the Search CLI package from this repo like so
    # this will install dependencies (ex: pandas) and small sample data sets 
(venv) $ python -m pip install git+https://github.com/sanjibg01/CS410-project-team-trailblazers.git
```

Example Usage with CLI App

Show help for the CLI app
```bash
# ready to go!
(venv) $ python -m search --help
Usage: python -m search [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  list-lectures   CLI command for displaying all lectures.
  query-arxiv     CLI command for querying arxiv papers
  query-lectures  CLI command for querying lectures
```

Show help for one command
```bash
(venv) $ python -m search query-arxiv --help
Usage: python -m search query-arxiv [OPTIONS] QUERY

  CLI command for querying arxiv papers

  Example usage: $ python -m search query-arxiv "natural language"

Options:
  --help  Show this message and exit.
```

List lectures. We use the 'text-retrieval' course by default, though we can also switch to 'text-mining'. 
```bash
(venv) $ python -m search list-lectures
ID	Lecture Title
0	01_course-welcome-video.en.txt
1	01_lesson-1-1-natural-language-content-analysis.en.txt
...
43	09_lesson-6-9-recommender-systems-collaborative-filtering-part-3.en.txt
44	10_lesson-6-10-course-summary.en.txt
```

Query lectures and display relevant matches.
```bash
(venv) $ python -m search query-lectures "vector space model"
QUERY: 'vector space model'
TOP 10 MATCHES IN LECTURE TRANSCRIPTS
|   document_id | title                                                                    |     score | document_preview                                   |
|--------------:|:-------------------------------------------------------------------------|----------:|:---------------------------------------------------|
|             6 | 05_lesson-1-5-vector-space-model-basic-idea.en.txt                       | 0.289408  | [SOUND] This lecture is about the vector space ret |
|             7 | 06_lesson-1-6-vector-space-retrieval-model-simplest-instantiation.en.txt | 0.27623   | In this lecture we're going to talk about how to i |
|            29 | 03_lesson-5-3-feedback-in-text-retrieval-feedback-in-lm.en.txt           | 0.215963  | [SOUND] This lecture is about the feedback in the  |
|             8 | 01_lesson-2-1-vector-space-model-improved-instantiation.en.txt           | 0.179016  | [SOUND] In this lecture, we are going to talk abou |
|            28 | 02_lesson-5-2-feedback-in-vector-space-model-rocchio.en.txt              | 0.171017  | [SOUND] This lecture is about the feedback in the  |
|            10 | 03_lesson-2-3-doc-length-normalization.en.txt                            | 0.148928  | [SOUND] This lecture is about Document Length Norm |
|            21 | 02_lesson-4-2-statistical-language-model.en.txt                          | 0.138494  | [SOUND] This lecture is about the statistical lang |
|            24 | 05_lesson-4-5-statistical-language-model-part-2.en.txt                   | 0.117617  | [SOUND] So I showed you how we rewrite the query l |
|            26 | 07_lesson-4-7-smoothing-methods-part-2.en.txt                            | 0.107794  | [SOUND] So let's plug in these model masses into t |
|            22 | 03_lesson-4-3-query-likelihood-retrieval-function.en.txt                 | 0.0890911 | [SOUND] This lecture is about query likelihood, pr |
```

Query Arxiv papers and display relevant matches.
```bash
(venv) $ python -m search query-lectures "vector space model"
QUERY: 'vector space model'
TOP 10 MATCHES IN ARXIV ABSTRACTS
PAPERS SEARCHED: 10000
|   document_id | title                                                                               |    score | document_preview                                 |
|--------------:|:------------------------------------------------------------------------------------|---------:|:-------------------------------------------------|
|          6996 | 0705.2994 - Non-relativistic limit of the Einstein equation                         | 0.353065 | In particular cases of stationary and stationary |
|          8558 | 0705.4556 - Quantization of symplectic vector spaces over finite fields             | 0.312484 | In this paper, we construct a quantization funct |
|          7336 | 0705.3334 - Supergravity inspired Vector Curvaton                                   | 0.29658  | It is investigated whether a massive Abelian vec |
|          1985 | 0704.1986 - Characterization of Closed Vector Fields in Finsler Geometry            | 0.251997 | The $\pi$-exterior derivative ${\o}d$, which is  |
|          7009 | 0705.3007 - A Generalization of Slavnov-Extended Non-Commutative Gauge Theories     | 0.244722 | We consider a non-commutative U(1) gauge theory  |
|          8577 | 0705.4575 - Higgsless Electroweak Theory following from the Spherical Geometry      | 0.242075 | A new formulation of the Electroweak Model with  |
|          9998 | 0706.1312 - Entrelacement d'alg\`ebres de Lie [Wreath products for Lie algebras]    | 0.232391 | Full details are given for the definition and co |
|           487 | 0704.0488 - Teleparallel Version of the Stationary Axisymmetric Solutions and their | 0.23176  | This work contains the teleparallel version of t |
|               |   Energy Contents                                                                   |          |                                                  |
|          6083 | 0705.2081 - On an identity for the volume integral of the square of a vector field  | 0.228912 | A proof is given of the vector identity proposed |
|           647 | 0704.0648 - Behavioral response to strong aversive stimuli: A neurodynamical model  | 0.219954 | In this paper a theoretical model of functioning |
```

```bash
(venv) $ python -m search query-arxiv "natural language processing"
QUERY: 'natural language processing'
TOP 10 MATCHES IN ARXIV ABSTRACTS
PAPERS SEARCHED: 10000
|   document_id | title                                                                                |    score | document_preview                                 |
|--------------:|:-------------------------------------------------------------------------------------|---------:|:-------------------------------------------------|
|          3369 | 0704.3370 - Natural boundary of Dirichlet series                                     | 0.250595 | We prove some conditions on the existence of nat |
|          5300 | 0705.1298 - Mykyta the Fox and networks of language                                  | 0.229746 | The results of quantitative analysis of word dis |
|          1318 | 0704.1319 - Using conceptual metaphor and functional grammar to explore how language | 0.22556  | This paper introduces a theory about the role of |
|               |   used in physics affects student learning                                           |          |                                                  |
|          3664 | 0704.3665 - On the Development of Text Input Method - Lessons Learned                | 0.221328 | Intelligent Input Methods (IM) are essential for |
|           690 | 0704.0691 - Birth, survival and death of languages by Monte Carlo simulation         | 0.219905 | Simulations of physicists for the competition be |
|            70 | 0704.0071 - Pairwise comparisons of typological profiles (of languages)              | 0.211776 | No abstract given; compares pairs of languages f |
|          7951 | 0705.3949 - Translating a first-order modal language to relational algebra           | 0.198326 | This paper is about Kripke structures that are i |
|          3885 | 0704.3886 - A Note on Ontology and Ordinary Language                                 | 0.183935 | We argue for a compositional semantics grounded  |
|          4397 | 0705.0395 - On logical characterization of henselianity                              | 0.1778   | We give some sufficient conditions under which a |
|          8305 | 0705.4303 - Database Manipulation on Quantum Computers                               | 0.155392 | Manipulating a database system on a quantum comp |
```

Example usage from Python. <br>This is equivalent to the CLI command: `$ python -m search query-arxiv "natural language"`
```python
from search import SearchEngine

query = "natural language"
se = SearchEngine()
se.query_lectures(query)
```
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

@paulzuradzki (Paul Zuradzki)
* `search/search.py` - TF-IDF and cosine similary implementation of search, command-line interface search app
* Python packaging
    * researched and implemented use of `__init__.py`, `setup.py`, and `importlib.resources` in order to deliver CLI app via pip
* Scraped lecture transcripts - pre-processed Coursera transcript files and merged across lectures
    * `search/data/`
    * `lecture_transcript_minining`
    * Used `coursera-dl` to webscrape and retrieve Coursera files
        * Involved non-trivial troubleshooting with browser cookie authentication and modifying source pacakage utils.py file 
    * `coursera-dl` splits files by week. Wrote code to collect transcript files from all lecture weeks and merge into one.
        * https://github.com/sanjibg01/CS410-project-team-trailblazers/blob/paulzuradzki/scripts/fetch_transcripts.py
* Documentation under heading "Search Command Line Interface" and associated CLI usage instructions
