import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV


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


def examine_tfidf_matrix(tfidf, tfidf_vectorizer):
    # place tf-idf values in a pandas data frame

    df = pd.DataFrame(
        tfidf[0].T.todense(),
        index=tfidf_vectorizer.get_feature_names_out(),
        columns=["tfidf"])

    print(df.sort_values(by=["tfidf"], ascending=False))


def lda_example_simple(doc_term_matrix):
    # Example of fitting LDA
    lda = LatentDirichletAllocation(
        n_components=4,
        random_state=42)
    lda.fit(doc_term_matrix)
    topic_results = lda.transform(doc_term_matrix)
    print(topic_results)


def lda_example_full(doc_term_matrix):
    # Example of fitting LDA
    lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                          max_iter=10,               # Max learning iterations
                                          learning_method='online',
                                          random_state=100,          # Random state
                                          batch_size=128,            # n docs in each learning iter
                                          evaluate_every=-1,         # compute perplexity every n iters, default: Don't
                                          n_jobs=-1,                 # Use all available CPUs
                                          )

    lda_output = lda_model.fit_transform(doc_term_matrix)
    return lda_model, lda_output


def lda_score(lda_model, doc_term_matrix):
    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(doc_term_matrix))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(doc_term_matrix))

    # See model parameters
    print(lda_model.get_params())


def grid_search_results_simple(doc_term_matrix):
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

    # Init the Model
    lda = LatentDirichletAllocation()

    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(doc_term_matrix)

    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(doc_term_matrix))

    return(best_lda_model)

n_topics = [10, 15, 20, 25, 30]

def grid_search_results_detail(model, n_topics):
    # Get Log Likelyhoods from Grid Search Output
    n_topics = [10, 15, 20, 25, 30]

    log_likelyhoods_5 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.5]
    log_likelyhoods_7 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.7]
    log_likelyhoods_9 = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in enumerate(model.cv_results_['params']) if gscore['learning_decay']==0.9]

    # Show graph
    plt.figure(figsize=(12, 8))
    plt.plot(n_topics, log_likelyhoods_5, label='0.5')
    plt.plot(n_topics, log_likelyhoods_7, label='0.7')
    plt.plot(n_topics, log_likelyhoods_9, label='0.9')
    plt.title("Choosing Optimal LDA Model")
    plt.xlabel("Num Topics")
    plt.ylabel("Log Likelyhood Scores")
    plt.legend(title='Learning decay', loc='best')
    plt.show()


best_lda_model = grid_search_results_simple(doc_term_matrix)

# Create Document - Topic Matrix
lda_output = best_lda_model.transform(doc_term_matrix)

# column names
topicnames = ["Topic" + str(i) for i in range(10)]  # n_topics in best model

# index names
docnames = ["Doc" + str(i) for i in range(len(corpus))]

print(lda_output)
print(len(topicnames))
print(len(docnames))

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic


# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


# Apply Style
# df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
print(df_document_topic.head())

# Review topics distribution across documents
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
print(df_topic_distribution)




