# https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
# https://www.kaggle.com/Cornell-University/arxiv
import json
from input.category_map import category_map
import nltk


class ArxivDatasetPreprocesser:
    def __init__(self):
        # TODO replace read on local machine to cloud storage
        # self.raw_data_path = 'input/arxiv-metadata-oai-snapshot.json'
        self.raw_data_path = 'input/example.json'
        self.ids = []
        self.category_codes = []
        self.categories = []
        self.titles = []
        self.abstracts = []
        self.update_dates = []

    def supply_line(self):
        with open(self.raw_data_path, 'r') as fh:
            for line in fh:
                yield json.loads(line)

    def load_data(self):
        for line in self.supply_line():
            self.ids.append(line['id'])

            category_code = self.preprocess_category_codes(line['categories'])
            self.category_codes.append(category_code)
            self.categories.append(self.lookup_category_for_code(category_code))  # noqa: E501

            self.titles.append(line['title'])
            self.abstracts.append(line['abstract'])
            self.update_dates.append(line['update_date'])

    def print_load_results(self):
        print('Number of ids parsed: {}'.format(len(self.ids)))
        print('Number of category_codes parsed: {}'.format(len(self.category_codes)))  # noqa: E501
        print('Number of categories parsed: {}'.format(len(self.categories)))
        print('Number of titles parsed: {}'.format(len(self.titles)))
        print('Number of abstracts parsed: {}'.format(len(self.abstracts)))
        print('Number of update_dates parsed: {}'.format(len(self.update_dates)))  # noqa: E501

    def preprocess_category_codes(self, category_code):
        '''
        There may be >1 category code. For now, take the first.
        TODO revisit this
        '''
        categories = str.split(' ')
        return categories[0]

    def lookup_category_for_code(self, category_code):
        return category_map[category_code]


def main():
    preprocessor = ArxivDatasetPreprocesser()
    preprocessor.load_data()
    preprocessor.lookup_category_for_codes()
    preprocessor.print_load_results()


if __name__:
    main()
