import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
import gensim
import gensim.corpora as corpora

from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV


class TopicModeler:
    def __init__(self, file):
        self.corpus = self.get_corpus(file)
        self.vectorizer = None
        self.matrix = None
        self.grid_search_results = None
        self.model = None
        self.X = None

    def get_corpus(self, file):
        with open(file, 'r') as f:
            corpus = json.load(f)
        return corpus

    def build_doc_term_matrix(self):
        self.vectorizer = CountVectorizer()

        # if wanting to use n-grams:
        # vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))

        self.X = self.vectorizer.fit_transform(self.corpus)
        # vocab = vectorizer.get_feature_names_out()  # list of distinct tokens
        doc_term_matrix = self.X.toarray()

        self.matrix = doc_term_matrix

    def build_tfidf_matrix(self):
        '''
        Out: tfidf_matrix in numpy sparse row format
        '''
        self.vectorizer = TfidfVectorizer(use_idf=True)
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def matrix_to_df(self):
        # Place tf-idf values in a pandas data frame
        df = pd.DataFrame(
            self.matrix[0].T.todense(),
            index=self.vectorizer.get_feature_names_out(),
            columns=["tfidf"])

        df = df.sort_values(by=["tfidf"], ascending=False)

        return df

    def get_matrix_shape(self):
        print(self.matrix.shape)

    def fit_lda(self):
        # Example of fitting LDA
        self.model = LatentDirichletAllocation(n_components=20,           # Number of topics
                                               max_iter=10,               # Max learning iterations
                                               learning_method='online',
                                               random_state=100,          # Random state
                                               batch_size=128,            # n docs in each learning iter
                                               evaluate_every=-1,         # compute perplexity every n iters, default: Don't
                                               n_jobs=-1,                 # Use all available CPUs
                                               )

        # lda_output = self.model.fit_transform(self.matrix)

    def print_lda_results(self):
        # Log Likelyhood: Higher the better
        print("Log Likelihood: ", self.model.score(self.matrix))

        # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
        print("Perplexity: ", self.model.perplexity(self.matrix))

        # See model parameters
        print(self.model.get_params())

    def perform_grid_search(self, n_topics=[10, 15, 20, 25, 30], learning_decay=[.5, .7, .9]):
        search_params = {'n_components': n_topics, 'learning_decay': learning_decay}

        # Init the Model
        lda = LatentDirichletAllocation()

        # Init Grid Search Class
        grid_search_model_obj = GridSearchCV(lda, param_grid=search_params)

        # Do the Grid Search
        grid_search_model_obj.fit(self.matrix)

        self.grid_search_results = grid_search_model_obj

    def set_model_to_optimum(self):
        self.model = self.grid_search_results.best_estimator_

    def print_grid_search_results(self):
        # Model Parameters
        print('\n')
        print('Printing grid search results...')
        print("Best Model's Params: ", self.grid_search_results.best_params_)

        # Log Likelihood Score
        print("Best Log Likelihood Score: ", self.grid_search_results.best_score_)

        # Perplexity
        print("Model Perplexity: ", self.model.perplexity(self.matrix))

    def plot_grid_search_results(self, n_topics=[10, 15, 20, 25, 30]):
        # Get Log Likelyhoods from Grid Search Output
        log_likelyhoods_5 = [round(self.grid_search_results.cv_results_['mean_test_score'][index])
            for index, gscore in enumerate(self.grid_search_results.cv_results_['params']) if gscore['learning_decay'] == 0.5]

        log_likelyhoods_7 = [round(self.grid_search_results.cv_results_['mean_test_score'][index])
             for index, gscore in enumerate(self.grid_search_results.cv_results_['params']) if gscore['learning_decay'] == 0.7]

        log_likelyhoods_9 = [round(self.grid_search_results.cv_results_['mean_test_score'][index])\
            for index, gscore in enumerate(self.grid_search_results.cv_results_['params']) if gscore['learning_decay'] == 0.9]

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

    def print_selected_model_details(self):
        print('\n')
        print('Printing detailed grid search results...')

        # Create Document - Topic Matrix
        doc_topic_matrix = self.model.transform(self.matrix)

        # column names
        num_topics_in_best_model = self.grid_search_results.best_params_['n_components']
        topicnames = ["Topic" + str(i) for i in range(num_topics_in_best_model)]

        # index names
        docnames = ["Doc" + str(i) for i in range(len(self.corpus))]

        print(doc_topic_matrix)
        print(len(topicnames))
        print(len(docnames))

        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(
            np.round(doc_topic_matrix, 2),
            columns=topicnames,
            index=docnames)

        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic['dominant_topic'] = dominant_topic

        # Apply Style
        print(df_document_topic.head())

        # Review topics distribution across documents
        df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
        df_topic_distribution.columns = ['Topic Num', 'Num Documents']
        print(df_topic_distribution)

    def print_top_keywords_in_topic(self):

        def show_topics_helper(n_words=20):
            keywords = np.array(self.vectorizer.get_feature_names_out())
            topic_keywords = []
            for topic_weights in self.model.components_:
                top_keyword_locs = (-topic_weights).argsort()[:n_words]
                topic_keywords.append(keywords.take(top_keyword_locs))
            return topic_keywords

        topic_keywords = show_topics_helper(n_words=15)

        # Topic - Keywords Dataframe
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
        print(df_topic_keywords)


# def lda_plot(vectorizer, best_lda_model, doc_term_matrix):
#     # run in notebook
#     panel = pyLDAvis.sklearn.prepare(best_lda_model, np.matrix(doc_term_matrix), vectorizer, mds='tsne')
#     panel
