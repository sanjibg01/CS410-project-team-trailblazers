# coursera_arxiv_search
### Description
Our project falls under the “Intelligent Learning Platform” theme. Our idea is to develop a method that links Coursera course content (via lecture video transcriptions) to academic journal articles. This will enrich the Coursera course content by allowing students to see how topics being taught have appeared in scientific literature, hence more ‘intelligent learning.’ To accomplish this, we will mine for topics in a dataset of academic papers and a dataset of lecture video transcriptions, applying NLP techniques either directly taught in CS410 or closely related. 

### Notes
* `coursera_arxiv_search` will be structured as a Python package. The module imports should be coordinated within that directory and callable from outside it.

### Links
* Arxiv / Kaggle data sets
    * https://www.kaggle.com/Cornell-University/arxiv
* Coursera-dl
    * https://github.com/coursera-dl/coursera-dl
    * Python package for downloading all coursera contents
    * the prepared transcript files for the two TIS 410 MOOCs are in `data/coursera_transcripts`
* GitHub remote repo
    * https://github.com/sanjibg01/CS410-project-team-trailblazers


# Setup
TODO: sample. Not implemented yet.
```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install git+https://github.com/sanjibg01/CS410-project-team-trailblazers.git
```

# Usage
TODO: this is just an example API. Not implemented yet.

### Python Usage
```python
from coursera_arxiv_search import SearchEngine

se = SearchEngine()
print(se.list_lectures())
se.get_related_papers(lecture_title='foo')

```

### Command-line Interface Usage
```bash
$ python -m coursera_arxiv_search --help
$ python -m coursera_arxiv_search --list-lectures
$ python -m coursera_arxiv_search --query-lectures "Natural Language Processing"
$ python -m coursera_arxiv_search --query-papers "Natural Language Processing"
$ python -m coursera_arxiv_search --lecture-name 03_1-3-natural-language-content-analysis-part-1 --get-related-papers
```

# Extensions
* extend to multiple Coursera courses for which the user has access
* integrate with `coursera-dl`

# Citations
* Third-party Python libraries
    * coursera-dl
    * argparse