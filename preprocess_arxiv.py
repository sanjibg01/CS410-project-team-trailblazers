# https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
# https://www.kaggle.com/Cornell-University/arxiv
import json
import spacy
import os

from input.category_map import category_map


class ArxivDatasetPreprocesser:
    def __init__(self):
        # TODO replace read on local machine to cloud storage
        # self.raw_data_path = 'input/arxiv-metadata-oai-snapshot.json'
        self.raw_data_path = 'input/example.json'  # for testing
        self.ids = []
        self.category_codes = []
        self.categories = []
        self.titles = []
        self.tokenized_titles = []
        self.cleaned_titles = []
        self.abstracts = []
        self.tokenized_abstracts = []
        self.cleaned_abstracts = []
        self.update_dates = []
        self.sp = spacy.load('en_core_web_sm')

    def supply_line(self):
        '''
        Generator for JSON objects, which represent individual papers.
        '''
        with open(self.raw_data_path, 'r') as fh:
            for line in fh:
                yield json.loads(line)

    def extract_data(self):
        for line in self.supply_line():
            self.ids.append(line['id'])

            category_code = self.preprocess_category_codes(line['categories'])
            self.category_codes.append(category_code)

            if category_code[0] is None:
                self.categories.append(category_code[1])
            else:
                self.categories.append(self.lookup_category_for_code(category_code))

            self.titles.append(line['title'])
            self.abstracts.append(line['abstract'])
            self.update_dates.append(line['update_date'])

    def print_extract_step_results(self):
        print('Number of ids extracted: {}'.format(len(self.ids)))
        print('Number of category_codes extracted: {}'.format(len(self.category_codes)))
        print('Number of categories extracted: {}'.format(len(self.categories)))
        print('Number of titles extracted: {}'.format(len(self.titles)))
        print('Number of abstracts extracted: {}'.format(len(self.abstracts)))
        print('Number of update_dates extracted: {}'.format(len(self.update_dates)))
        print('\n')

    def preprocess_category_codes(self, category_code):
        '''
        There may be >1 category code. For now, take the first.
        TODO revisit this
        '''
        categories = str.split(' ')
        if categories:
            return categories[0]
        else:
            return (None, "No category provided")

    def lookup_category_for_code(self, category_code):
        return category_map[category_code]

    def clean_doc(self, doc):
        cleaned_doc = []

        sp_doc = self.sp(doc)  # tokenize
        for token in sp_doc:
            if not token.is_stop:  # remove stop words
                if token.is_alpha:  # remove numbers etc.
                    if token.pos_ == 'NOUN':
                        token.lemma_ = token.lemma_.lower()  # lemmaize and force lowercase
                        cleaned_doc.append(token.lemma_)

        return cleaned_doc

    def transform_titles(self):
        for title in self.titles:
            cleaned_title = self.clean_doc(title)
            self.tokenized_titles.append(cleaned_title)

    def transform_abstracts(self):
        for abstract in self.abstracts:
            cleaned_abstract = self.clean_doc(abstract)
            self.tokenized_abstracts.append(cleaned_abstract)

    def print_transform_step_results(self):
        # TODO if not match, raise err
        print('Number of ids extracted: {}'.format(len(self.ids)))
        print('Number of cleaned titles: {}'.format(len(self.cleaned_titles)))
        print('Number of cleaned abstracts: {}'.format(len(self.cleaned_abstracts)))
        print('\n')

    def reassemble_corpus(self, tokenized_corpus, cleaned_corpus):
        '''
        Using the preprocessed tokens, reassemble the original
        corpus.
        '''
        for doc in tokenized_corpus:
            cleaned_corpus.append(' '.join(doc))

    def write_output(self, filename, obj_to_write):
        with open(filename + '.json', 'w') as f:
            json.dump(obj_to_write, f, indent=2)


def main():
    # extract
    preprocessor = ArxivDatasetPreprocesser()
    preprocessor.extract_data()
    preprocessor.print_extract_step_results()

    # transform
    preprocessor.transform_titles()
    preprocessor.transform_abstracts()

    preprocessor.reassemble_corpus(preprocessor.tokenized_titles, preprocessor.cleaned_titles)
    preprocessor.reassemble_corpus(preprocessor.tokenized_abstracts, preprocessor.cleaned_abstracts)

    preprocessor.print_transform_step_results()

    # load
    preprocessor.write_output('preprocessed_abstracts', preprocessor.cleaned_abstracts)



if __name__:
    main()
