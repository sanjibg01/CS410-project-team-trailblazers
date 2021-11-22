import json
import spacy
import time
import argparse

from input.category_map import category_map


class DataPreprocessor:
    def __init__(self, file):
        self.raw_data_path = file
        self.ids = []
        self.category_codes = []
        self.categories = []
        self.tokenized_categories = []
        self.titles = []
        self.tokenized_titles = []
        self.abstracts = []
        self.tokenized_abstracts = []
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

    def print_extract_step_results(self):
        print('Number of ids extracted: {}'.format(len(self.ids)))
        print('Number of category_codes extracted: {}'.format(len(self.category_codes)))
        print('Number of categories extracted: {}'.format(len(self.categories)))
        print('Number of titles extracted: {}'.format(len(self.titles)))
        print('Number of abstracts extracted: {}'.format(len(self.abstracts)))
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

    def clean_doc(self, doc, keep_only_nouns=True, ngrams=False):
        # TODO refactor this
        cleaned_doc = []

        if ngrams:
            sp_doc = self.sp(doc)
            for chunk in sp_doc.noun_chunks:
                cleaned_chunk = ''
                for token in chunk:
                    if not token.is_stop and token.is_alpha:
                        cleaned_chunk = cleaned_chunk + str(token) + ' '
                tmp = cleaned_chunk.strip()
                if tmp != '':
                    cleaned_doc.append(tmp)

        else:
            sp_doc = self.sp(doc, disable=["parser", "ner"])  # tokenize
            if sp_doc is not None:
                for token in sp_doc:
                    if not token.is_stop:  # remove stop words
                        if token.is_alpha:  # remove numbers etc.
                            token.lemma_ = token.lemma_.lower()  # lemmaize and force lowercase

                            if (keep_only_nouns) & (token.pos_ == 'NOUN'):  # only nouns
                                cleaned_doc.append(token.lemma_)
                            else:
                                cleaned_doc.append(token.lemma_)

        return cleaned_doc

    def transform_titles(self, ngrams):
        for title in self.titles:
            if title is not None:
                cleaned_title = self.clean_doc(title, False, ngrams)
            self.tokenized_titles.append(cleaned_title)

    def transform_abstracts(self, ngrams):
        # start = time.time()
        for abstract in self.abstracts:
            if abstract is not None:
                cleaned_abstract = self.clean_doc(abstract, False, ngrams)
                self.tokenized_abstracts.append(cleaned_abstract)
                # end = time.time()
                # print(end - start)

    def transform_categories(self, ngrams):
        for category in self.categories:
            if category is not None:
                tokenized_category = self.clean_doc(category, False, ngrams)
                self.tokenized_categories.append(tokenized_category)

    def print_transform_step_results(self):
        # TODO if not match, raise err
        print('Number of ids extracted: {}'.format(len(self.ids)))
        print('Number of cleaned titles: {}'.format(len(self.tokenized_titles)))
        print('Number of cleaned abstracts: {}'.format(len(self.tokenized_abstracts)))
        print('\n')

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
        with open(filename, 'w') as f:
            json.dump(obj_to_write, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='file name, should be .json', required=True)
    parser.add_argument('--output_file', help='file name to create with output, should be .json', required=True)
    parser.add_argument('--ngrams', help='do you want unigrams or ngrams (chunks, usually 2 or 3) in result', required=False, default=False)
    args = parser.parse_args()

    # extract
    preprocessor = DataPreprocessor(args.input_file)
    preprocessor.extract_data()
    preprocessor.print_extract_step_results()

    # transform
    preprocessor.transform_titles(args.ngrams)
    print('Completed cleaning titles')
    preprocessor.transform_abstracts(args.ngrams)
    print('Completed cleaning abstracts')
    preprocessor.transform_categories(args.ngrams)
    print('Completed cleaning categories')

    preprocessor.print_transform_step_results()
    output = preprocessor.combine_output()

    # load
    preprocessor.write_output(args.output_file, output)


if __name__:
    main()
