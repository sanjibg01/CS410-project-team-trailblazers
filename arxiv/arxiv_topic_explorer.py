import json
import pandas as pd


class ArxivTopicExplorer:
    def __init__(self,
                 doc_file='input/input.json',
                 topics_file='output/document_indices_ranked.csv'):
        self.doc_file = doc_file
        self.topics_file = topics_file
        self.texts = []
        self.topics = None
        self.current_place = {}

    def set_texts(self):
        with open(self.doc_file, 'r') as fh:
            for line in fh:
                self.texts.append(json.loads(line))

    def set_topics(self):
        topics_df = pd.read_csv(self.topics_file)
        self.topics = topics_df.to_dict()

    def prompt(self):
        list_of_topics = list(self.topics)

        for topic_num, topic in enumerate(list_of_topics, start=1):
            doc_index = self.topics[topic][0]  # first topic
            self.get_article(topic_num, doc_index)
            self.increment_place(topic_num)

    def get_topics(self, topic_num):
        try:
            cur = self.current_place[topic_num]
        except KeyError:
            print('\nUnrecognized topic number. Try again.')
            return

        user_selected_topic_name = 'Topic_{}'.format(topic_num)

        for i in range(2):  # get three more articles
            try:
                doc_index = self.topics[user_selected_topic_name][cur]
            except KeyError:
                print('\nOut of journal articles for that topic!')
                return
            self.get_article(topic_num, doc_index)
            self.increment_place(topic_num)

    def get_article(self, topic_num, doc_index):
        print('\n')
        print('Article Group Number: {}'.format(topic_num))
        print('Article Title: ' + self.texts[doc_index]['title'].strip())
        print('Article Abstract Preview: ' + self.texts[doc_index]['abstract'].strip()[0:200] + '...')

    def increment_place(self, topic_num):
        if topic_num in self.current_place.keys():
            self.current_place[topic_num] += 1
        else:
            self.current_place[topic_num] = 1


if __name__ == '__main__':
    topic_explorer = ArxivTopicExplorer()
    topic_explorer.set_texts()
    topic_explorer.set_topics()

    print('Welcome to the interactive arXiv article explorer.')

    print('Below are a few journal articles...')
    topic_explorer.prompt()
    user_selected_topic = None

    while user_selected_topic != 'exit':
        user_selected_topic = input('Enter the number of an article that interests you to see additional recommended articles. \n Or enter \'exit\' to exit the program.')
        if user_selected_topic == 'exit':
            exit()
        else:
            topic_explorer.get_topics(int(user_selected_topic))
