import json
import os
from pathlib import Path
import metapy
import csv
# wget -nc https://raw.githubusercontent.com/meta-toolkit/meta/master/data/lemur-stopwords.txt
import pandas as pd
from gensim import corpora
import gensim
import matplotlib.pyplot as plt
import random
import numpy as np
random.seed(123)

def preprocess_text(doc):
    """ 
    Write token stream that:
    - tokenizes with ICUTokenizer, 
    - lowercases, 
    - removes words with less than 2 and more than 30  characters, and stop words
    - performs stemming and creates unigrams 
    
    """
    
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    tok = metapy.analyzers.LowercaseFilter(tok)
    tok = metapy.analyzers.LengthFilter(tok, min=2, max=30)
    tok = metapy.analyzers.Porter2Filter(tok)
    tok = metapy.analyzers.ListFilter(tok, "lemur-stopwords.txt", metapy.analyzers.ListFilter.Type.Reject)
    ana = metapy.analyzers.NGramWordAnalyzer(1, tok)

    processedtxt = ana.analyze(doc)
    
    tok.set_content(doc.content())
    tokens, counts = [], []
    for token, count in processedtxt.items():
        counts.append(count)
        tokens.append(token)
    return tokens, counts


def get_transcripts(filenames):
    """
    Load and combine transcripts in json format into one dictionary, with keys as lecture titles and values as transcript text.
    """
    
    all_lesson_titles = list()
    all_transcripts = {}
    for filename in filenames:
        
        with open(filename, 'r') as f:
            transcripts = json.load(f)
        
        all_transcripts.update(transcripts)
        
        lesson_titles = list(transcripts.keys())
        all_lesson_titles = all_lesson_titles + lesson_titles
 
    return all_transcripts, all_lesson_titles

def get_tokens(transcripts, lesson_titles):
    """
    Pre-process each document into list of tokens (calling preprocess_text()).
    """
    
    lecture_tokens = {}
    lecture_token_counts = {}
    lecture_tokens_list = []
    for title in lesson_titles: 

        doctxt = transcripts[title]

        doc = metapy.index.Document()
        doc.content(doctxt)

        tokens,counts = preprocess_text(doc)
        lecture_tokens[title] = tokens
        lecture_token_counts[title] = counts
        lecture_tokens_list.append(tokens)
    return lecture_tokens, lecture_tokens_list, lecture_token_counts

def output_tokens_tocsv(lecture_tokens, output_filename):
    
    with open(output_filename, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in lecture_tokens.items():
           writer.writerow([key, value])

def create_lda_model(lecture_tokens_list, ntopics, niterations):
    """
    Transform tokens data to BoW with Tf-IDF weighting to consider rare terms more while reducing weight on common words (important especially for verbal text).
    Create LDA model with given parameters found through model grid search and experimentation.
    Code adapted from: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
                       https://highdemandskills.com/topic-modeling-lda/
    """
    
    np.random.seed(12345)
    dictionary = corpora.Dictionary(lecture_tokens_list)
    corpus = [dictionary.doc2bow(text) for text in lecture_tokens_list]

    TFIDF = gensim.models.TfidfModel(corpus) 
    trans_TFIDF = TFIDF[corpus] 

    best_model = gensim.models.ldamodel.LdaModel(trans_TFIDF, 
                                                   num_topics = ntopics, 
                                                   id2word=dictionary,
                                                   iterations = niterations
                                                  )
    return best_model, corpus


def find_best_model_gridsearch(lecture_tokens_list):
    """
    Perform grid search through number of topics to find optimal model in terms of topic coherence values.
    Visualize plot on parameters and their performance.
    Get best model with highest performance.
    
    """
    dictionary = corpora.Dictionary(lecture_tokens_list)
    corpus = [dictionary.doc2bow(text) for text in lecture_tokens_list]

    TFIDF = gensim.models.TfidfModel(corpus) # Fit TF-IDF model
    trans_TFIDF = TFIDF[corpus] # Apply TF-IDF model

    start = 3
    stop = 30
    step = 2
    coherence_values = []
    candidate_models = []

    for n_topics in range(start, stop, step):
            ldamodel = gensim.models.ldamodel.LdaModel(trans_TFIDF, 
                                                       num_topics = n_topics, 
                                                       id2word=dictionary,
                                                       iterations = 200
                                                      )
            coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel, texts=lecture_tokens_list, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherence_model_lda.get_coherence())
            candidate_models.append((n_topics, ldamodel))

    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    # get best topic model
    best_model_coherence = max(coherence_values)
    best_model_index = coherence_values.index(max(coherence_values))
    best_model_ntopics, best_model = candidate_models[best_model_index]
    print('# topics: ', best_model_ntopics, best_model_coherence)
    return best_model

def get_doc_summary(model, lecture_tokens_list, corpus,lesson_titles): 
    """
    Get the most dominant topic for each doc, and their topic coverage, top keywords in dominant topic.
    Adapted from arxiv/topic_modeler.py
    """
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(model[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([lesson_titles[i],int(topic_num), round(prop_topic, 4), row, topic_keywords]), ignore_index=True)
                else:
                    break
    sent_topics_df.columns = ['Lecture_Title','Dominant_Topic', 'Dominant_Topic_Perc_Contribution','Topic Coverage', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(lecture_tokens_list)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    doc_topic_summary = sent_topics_df.reset_index()
    doc_topic_summary.columns = ['Document_No','Lecture_Title', 'Dominant_Topic', 'Dominant_Topic_Perc_Contribution','Topic Coverage', 'Keywords', 'Text']

    return sent_topics_df, doc_topic_summary

 
def produce_topic_summary_df(sent_topics_df): 
    """
    Produce topic distribution across documents to assess topic selection quality.
    Adapted from arxiv/topic_modeler.py
    """
        
    topic_summary = sent_topics_df.groupby(['Dominant_Topic', 'Topic_Keywords']).agg('count').reset_index().sort_values('Dominant_Topic_Perc_Contribution', ascending=False)
    topic_summary = topic_summary.iloc[: , :-3]
    topic_summary.columns = ['Dominant_Topic', 'Topic_Keywords','Num_Documents']
    topic_summary['Perc_Documents'] = topic_summary.Num_Documents/topic_summary.Num_Documents.sum()
    topic_summary

    return topic_summary
 
def main():
    
    filenames = ['transcripts_text-mining_txt.json','transcripts_text-retrieval_txt.json']
    transcripts, lesson_titles = get_transcripts(filenames)
    lecture_tokens, lecture_tokens_list, lecture_token_counts = get_tokens(transcripts, lesson_titles)    
    optmodel = find_best_model_gridsearch(lecture_tokens_list)    
    model,corpus = create_lda_model(lecture_tokens_list, ntopics = 10, niterations = 200)
    sent_topics_df, doc_topic_summary = get_doc_summary(model, lecture_tokens_list, corpus,lesson_titles)
    
    topic_summary = produce_topic_summary_df(sent_topics_df)
    print(topic_summary)
    print(doc_topic_summary)
    output_filename = 'doc_topic_summary.csv'
    print('Topic Data output file: ', output_filename)
    doc_topic_summary.to_csv(output_filename)
    


if __name__:
    main()

    
