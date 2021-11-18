import json
import spacy
import time

from input.category_map import category_map


class ArxivDatasetPreprocesser:
    def __init__(self):
        # self.raw_data_path = 'input/arxiv-metadata-oai-snapshot.json'
        self.raw_data_path = 'input/example2.json'  # for testing
        self.ids = []
        self.category_codes = []
        self.categories = []
        self.tokenized_categories = []
        self.cleaned_categories = []
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

            if 'cs' not in line['categories']:
                continue

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
        categories = category_code.split(' ')
        if categories:
            return categories[0]
        else:
            return (None, "No category provided")

    def lookup_category_for_code(self, category_code):
        if category_code in category_map:
            return category_map[category_code]

    def clean_doc(self, doc, keep_only_nouns=True):
        cleaned_doc = []

        sp_doc = self.sp(doc, disable=["parser", "ner"])  # tokenize
        for token in sp_doc:
            if not token.is_stop:  # remove stop words
                if token.is_alpha:  # remove numbers etc.
                    token.lemma_ = token.lemma_.lower()  # lemmaize and force lowercase
                    if keep_only_nouns:
                        if token.pos_ == 'NOUN':
                            cleaned_doc.append(token.lemma_)
                    else:
                        cleaned_doc.append(token.lemma_)

        return cleaned_doc

    def transform_titles(self):
        for title in self.titles:
            cleaned_title = self.clean_doc(title)
            self.tokenized_titles.append(cleaned_title)

    def transform_abstracts(self):
        # start = time.time()
        for abstract in self.abstracts:
            cleaned_abstract = self.clean_doc(abstract)
            self.tokenized_abstracts.append(cleaned_abstract)
            # end = time.time()
            # print(end - start)

    def transform_categories(self):
        for category in self.categories:
            tokenized_category = self.clean_doc(category, keep_only_nouns=False)
            self.tokenized_categories.append(tokenized_category)

    def print_transform_step_results(self):
        # TODO if not match, raise err
        print('Number of ids extracted: {}'.format(len(self.ids)))
        print('Number of cleaned titles: {}'.format(len(self.tokenized_titles)))
        print('Number of cleaned abstracts: {}'.format(len(self.tokenized_abstracts)))
        print('\n')

    def reassemble_corpus(self, tokenized_corpus, cleaned_corpus):
        '''
        Using the preprocessed tokens, reassemble the original
        corpus.
        '''
        for doc in tokenized_corpus:
            cleaned_corpus.append(' '.join(doc))

    def combine_output(self):
        output = []

        for abstract, title, category in zip(self.tokenized_abstracts, self.tokenized_titles, self.tokenized_categories):
            # start = time.time()
            doc = []
            doc.extend(abstract)

            if title != '':
                doc.extend(title)

            if category != 'no category provided':
                doc.extend(category)

            output.append(doc)
            # end = time.time()
            # print(end - start)

        return output

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
    print('Completed cleaning titles')
    preprocessor.transform_abstracts()
    print('Completed cleaning abstracts')
    preprocessor.transform_categories()
    print('Completed cleaning categories')

    # preprocessor.reassemble_corpus(preprocessor.tokenized_titles, preprocessor.cleaned_titles)
    # print('Completed reassembling titles')
    # preprocessor.reassemble_corpus(preprocessor.tokenized_abstracts, preprocessor.cleaned_abstracts)
    # print('Completed reassembling abstracts')
    # preprocessor.reassemble_corpus(preprocessor.tokenized_categories, preprocessor.cleaned_categories)
    # print('Completed reassembling categories')

    preprocessor.print_transform_step_results()
    output = preprocessor.combine_output()

    # load
    preprocessor.write_output('output', output)


if __name__:
    main()
