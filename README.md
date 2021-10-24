# Description
Our project falls under the “Intelligent Learning Platform” theme. Our idea is to develop a method that links Coursera course content (via lecture video transcriptions) to academic journal articles. This will enrich the Coursera course content by allowing students to see how topics being taught have appeared in scientific literature, hence more ‘intelligent learning.’ To accomplish this, we will mine for topics in a dataset of academic papers and a dataset of lecture video transcriptions, applying NLP techniques either directly taught in CS410 or closely related. 

# Notes
* `coursera_arxiv_search` will be structured as a Python package. The module imports should be coordinated within that directory and callable from outside it.

# Links
* Arxiv / Kaggle data sets
    * https://www.kaggle.com/Cornell-University/arxiv
* Coursera-dl
    * https://github.com/coursera-dl/coursera-dl
    * Python package for downloading all coursera contents
    * the prepared transcript files for the two TIS 410 MOOCs are in `data/coursera_transcripts`