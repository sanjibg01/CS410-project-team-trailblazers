import json


class ArxivTopicExplorer:
    def __init__(self, file='output/arxiv_with_topics.json'):
        self.file = file
        self.texts = None

    def set_texts(self):
        with open(self.file, 'r') as f:
            self.texts = json.load(f)

    def get_topics(self):
        pass


if __name__ == '__main__':
    topic_explorer = ArxivTopicExplorer()
    print('Welcome to the interactive arXiv topic explorer')

    user_selected_topic = input('Select one of the following topics:')
    ArxivTopicExplorer.get_topics()

    print('The following jounral articles may be of interest to you:')

    while user_selected_topic != 'exit':
        user_selected_topic = input('Select another topic, or enter \'exit\' to exit the arXiv topic explorer:')
        if user_selected_topic == 'exit':
            exit()
        else:
            ArxivTopicExplorer.get_topics()
