import json
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation


with open('preprocessed_abstracts.json', 'r') as f:
    corpus = json.load(f)

vectorizer = CountVectorizer()

# if wanting to use n-grams:
# vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))

X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()  # list of distinct tokens
doc_term_matrix = X.toarray()


# tf-idf
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
tfidf = tfidf_vectorizer.fit_transform(corpus)


# place tf-idf values in a pandas data frame
df = pd.DataFrame(
    tfidf[0].T.todense(),
    index=tfidf_vectorizer.get_feature_names_out(),
    columns=["tfidf"])

print(df.sort_values(by=["tfidf"], ascending=False))
