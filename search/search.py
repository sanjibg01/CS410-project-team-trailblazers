"""search.py

Description: The `search` module implements search using TF-IDF and cosine similarity. We can search Coursera lecture transcripts and/or Arxiv research papers. 

Usage: python -m search [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  query-arxiv     CLI command for querying arxiv papers
  query-lectures  CLI command for querying lectures

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from pathlib import Path
import json
from pprint import pprint
import click
import pkgutil
from importlib.resources import read_text

pd.set_option("max_colwidth", 75)
pd.set_option("max_columns", 100)


class SearchEngine:
    def __init__(
        self,
        course_transcript_filename="transcripts_text-retrieval_txt.json",
        limit_arxiv_papers=True,
    ):
        self.limit_arxiv_papers = limit_arxiv_papers
        self.course_transcript_filename = course_transcript_filename
        self.arxiv_data = self.load_arxiv_data()
        self.lecture_data = self.load_lecture_data()

    def load_arxiv_data(self):

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

        textdata = read_text("search.data", self.course_transcript_filename)
        lecture_data = json.loads(textdata)
        for title, doc in lecture_data.items():
            lecture_data[title] = doc.replace("\n", " ")
        return lecture_data

    def list_lectures(self):
        print("ID\tLecture Title")
        for idx, lecture_title in enumerate(sorted(self.lecture_data.keys())):
            print(f"{idx}\t{lecture_title}")

    def query_lectures(self, query):
        documents: List[str] = list(self.lecture_data.values())
        tfidf_search = TfidfCosineSearch(query, documents)
        scored_docs = tfidf_search.score_documents()
        scored_docs["title"] = scored_docs["document_id"].map(
            dict(enumerate(self.lecture_data.keys()))
        )
        scored_docs = scored_docs[["document_id", "title", "score", "document"]]
        scored_docs["document_preview"] = scored_docs["document"].str.slice(0, 50)
        top_n_docs = 10
        print(f"QUERY: '{query}'")
        print(f"TOP {top_n_docs} MATCHES IN LECTURE TRANSCRIPTS")
        print(
            scored_docs.loc[:, ["document_id", "title", "score", "document_preview"]]
            .head(top_n_docs)
            .to_markdown(index=False)
        )

    def query_arxiv(self, query):
        documents: List[str] = list(self.arxiv_data.values())
        tfidf_search = TfidfCosineSearch(query, documents)
        scored_docs = tfidf_search.score_documents()
        scored_docs["title"] = scored_docs["document_id"].map(
            dict(enumerate(self.arxiv_data.keys()))
        )
        scored_docs = scored_docs[["document_id", "title", "score", "document"]]
        scored_docs["document_preview"] = scored_docs["document"].str.slice(0, 50)
        top_n_docs = 10
        print(f"QUERY: '{query}'")
        print(f"TOP {top_n_docs} MATCHES IN ARXIV ABSTRACTS")
        print(f"# PAPERS SEARCHED: {self.limit_arxiv_papers}")
        print(
            scored_docs.loc[:, ["document_id", "title", "score", "document_preview"]]
            .head(top_n_docs)
            .to_markdown(index=False)
        )


class TfidfCosineSearch:
    def __init__(self, query, documents):
        self.query = query
        self.documents = documents
        self.scores_df = self.score_documents()

    def make_tfidf_weights_df(self):
        query_and_docs = [self.query] + self.documents
        vectorizer = TfidfVectorizer()
        transformed = vectorizer.fit_transform(query_and_docs)
        tfidf_as_df = pd.DataFrame(
            data=transformed.toarray(), columns=vectorizer.get_feature_names_out()
        )
        tfidf_as_df.index = tfidf_as_df.index - 1
        tfidf_as_df.index.name = "document_id"

        return tfidf_as_df.replace(0, "")

    def score_documents(self):
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
    query()
