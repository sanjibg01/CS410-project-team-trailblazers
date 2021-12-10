"""search.py

Description: The `search` module implements search using TF-IDF and cosine similarity. We can search Coursera lecture transcripts and/or Arxiv research papers. 

See README.md for detailed examples.

Usage: python -m search [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  query-arxiv     CLI command for querying arxiv papers
  query-lectures  CLI command for querying lectures

"""
import json
import pkgutil
from importlib.resources import read_text
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import click
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("max_colwidth", 75)
pd.set_option("max_columns", 100)


class SearchEngine:
    """Implements query-document matching via TF-IDF weights and cosine similarity.

    Includes methods for loading Arxiv paper data (sample), loading Coursera lecture transcripts, and performing the queries.
    """

    def __init__(
        self,
        course_transcript_filename="transcripts_text-retrieval_txt.json",
        limit_arxiv_papers=True,
    ):
        self.limit_arxiv_papers = limit_arxiv_papers
        self.course_transcript_filename = course_transcript_filename
        self.arxiv_data = self.load_arxiv_data()
        self.lecture_data = self.load_lecture_data()

    def load_arxiv_data(self) -> Dict:
        """Loads Arxiv research paper documents from ./search/data/arxiv-small.json"""

        # use pkgutil to use package data
        # since we are distributing as a package, it is not as straightforward as using `with open()...` from a filepath
        textdata = read_text("search.data", "arxiv-small.json").split("\n")[:-1]

        # arxiv_data_processed is in format of {'id_title': '<abstract>', ..., 'id_title': '<abstract>'}
        arxiv_data = {}
        for line in textdata:
            doc = json.loads(line)
            id_title = "{} - {}".format(doc["id"], doc["title"])
            arxiv_data[id_title] = doc["abstract"]

        return arxiv_data

    def load_lecture_data(self):
        """Loads Coursera lecture data.

        An alternative course title file can be provided to extend this.
        For now, our two course options are 'transcripts_text-retrieval_txt.json' or 'transcripts_text-mining_txt.json' (retrieval and mining).
        """

        textdata = read_text("search.data", self.course_transcript_filename)
        lecture_data = json.loads(textdata)
        for title, doc in lecture_data.items():
            lecture_data[title] = doc.replace("\n", " ")
        return lecture_data

    def list_lectures(self):
        """Displays all lecture titles.

        This is a helper to guide a user in search terms they may use in a query.
        """

        print("ID\tLecture Title")
        for idx, lecture_title in enumerate(sorted(self.lecture_data.keys())):
            print(f"{idx}\t{lecture_title}")

    def query_lectures(self, query, n_docs=10):
        """Prints top N list of lectures that match a query in best-to-worst order."""

        documents: List[str] = list(self.lecture_data.values())
        tfidf_search = TfidfCosineSearch(query, documents)
        scored_docs = tfidf_search.make_document_scores_df()
        scored_docs["title"] = scored_docs["document_id"].map(
            dict(enumerate(self.lecture_data.keys()))
        )
        scored_docs = scored_docs[["document_id", "title", "score", "document"]]
        scored_docs["document_preview"] = scored_docs["document"].str.slice(0, 50)
        top_n_docs = n_docs
        print(f"QUERY: '{query}'")
        print(f"TOP {top_n_docs} MATCHES IN LECTURE TRANSCRIPTS")
        print(
            scored_docs.loc[:, ["document_id", "title", "score", "document_preview"]]
            .head(top_n_docs)
            .to_markdown(index=False)
        )

    def query_arxiv(self, query, n_docs=10):
        """Prints top N list of Arxiv papers that match a query in best-to-worst order."""

        documents: List[str] = list(self.arxiv_data.values())
        tfidf_search = TfidfCosineSearch(query, documents)
        scored_docs = tfidf_search.make_document_scores_df()
        scored_docs["title"] = scored_docs["document_id"].map(
            dict(enumerate(self.arxiv_data.keys()))
        )
        scored_docs = scored_docs[["document_id", "title", "score", "document"]]
        scored_docs["document_preview"] = scored_docs["document"].str.slice(0, 50)
        top_n_docs = n_docs
        print(f"QUERY: '{query}'")
        print(f"TOP {top_n_docs} MATCHES IN ARXIV ABSTRACTS")
        print(
            f"NUMBER OF PAPERS SEARCHED: 10,000"
        )  # FIXME: hard-coded to 10,000. This should be changed if we extend the model to a larger data source.
        print(
            scored_docs.loc[:, ["document_id", "title", "score", "document_preview"]]
            .head(top_n_docs)
            .to_markdown(index=False)
        )


class TfidfCosineSearch:
    """Implements query-document matching given a query and document.

    This class is generalized to be agnostic of the data domain.
    """

    def __init__(self, query, documents):
        self.query = query
        self.documents = documents
        self.scores_df = self.make_document_scores_df()

    def make_tfidf_weights_df(self) -> pd.DataFrame:
        """Produces formatted dataframe of TF-IDF weights for query and documents."""
        query_and_docs = [self.query] + self.documents
        vectorizer = TfidfVectorizer()
        transformed = vectorizer.fit_transform(query_and_docs)
        tfidf_as_df = pd.DataFrame(
            data=transformed.toarray(), columns=vectorizer.get_feature_names_out()
        )
        tfidf_as_df.index = tfidf_as_df.index - 1
        tfidf_as_df.index.name = "document_id"

        return tfidf_as_df.replace(0, "")

    def make_document_scores_df(self):
        """Dataframe with document scores (scoring relevance to query).

        This dataframe can be used for analysis.
        From a CLI-user's POV, the dataframe is later displayed as terminal output limiting top N documents and showing document preview only.
        This returned dataframe is useful for inspecting the full document without cross-refencing to source data.
        """

        query_and_docs = [self.query] + self.documents
        vectorizer = TfidfVectorizer()
        transformed = vectorizer.fit_transform(query_and_docs)

        query_weights = transformed[0]
        document_weights_all = transformed[1:]
        scores = cosine_similarity(query_weights, document_weights_all)

        # sort descending by score/weight
        scores = sorted(list(enumerate(scores[0])), key=lambda x: x[1], reverse=True)

        # adding document so we have more than just the document index
        scores = [
            (doc_index, self.documents[doc_index], weight)
            for doc_index, weight in scores
        ]
        scores_df = pd.DataFrame(scores, columns=["document_id", "document", "score"])

        return scores_df


#######################################
# Command Line Interface Specification
#######################################
@click.group()
def query():
    pass


@query.command()
@click.argument("query")
def query_lectures(query):
    """CLI command for querying lectures

    Example usage: $ python -m search query-arxiv "natural language"
    """
    se = SearchEngine()
    se.query_lectures(query)


@query.command()
@click.argument("query")
def query_arxiv(query):
    """CLI command for querying arxiv papers

    Example usage: $ python -m search query-arxiv "natural language"
    """

    se = SearchEngine()
    se.query_arxiv(query)


@query.command()
def list_lectures():
    """CLI command for displaying all lectures.

    Example usage: $ python -m search list-lectures
    """

    se = SearchEngine()
    se.list_lectures()


if __name__ == "__main__":
    # initializes CLI
    query()
