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

    def clean_doc(self, doc, keep_only_nouns=True, bigrams=False):
        # TODO refactor this
        cleaned_doc = []

        # TODO remove stop words, something like this:
        #         noun_phrases = []
        # for chunk in docx.noun_chunks:
        #     print(chunk)
        #     if all(token.is_stop != True and token.is_punct != True and '-PRON-' not in token.lemma_ for token in chunk) == True:
        #         if len(chunk) > 1:
        #             noun_phrases.append(chunk)
        # print(noun_phrases)

        if bigrams:
            sp_doc = self.sp(doc)
            for token in sp_doc.noun_chunks:
                cleaned_doc.append(token.text.lower().strip())

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

    def transform_titles(self, bigrams):
        for title in self.titles:
            cleaned_title = self.clean_doc(title, False, bigrams)
            self.tokenized_titles.append(cleaned_title)

    def transform_abstracts(self, bigrams):
        # start = time.time()
        for abstract in self.abstracts:
            cleaned_abstract = self.clean_doc(abstract, False, bigrams)
            self.tokenized_abstracts.append(cleaned_abstract)
            # end = time.time()
            # print(end - start)

    def transform_categories(self, bigrams):
        for category in self.categories:
            tokenized_category = self.clean_doc(category, False, bigrams)
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
        with open(filename + '.json', 'w') as f:
            json.dump(obj_to_write, f, indent=2)

    def bigram(doc):
        # Attribute: https://github.com/EricFillion/N-Grams/blob/master/ngrams.py

        # create a list for the result
        result = list()

        # create a list that contains no punctuation
        sentence = list()

        # parse through the document to add all tokens that are words to the sentence list
        for token in doc:
            if token.is_alpha:
                sentence.append(token)
        # parse through the sentence while adding words in groups of two to the result
        for word in range(len(sentence) - 1):
            first_word = sentence[word]
            second_word = sentence[word + 1]
            element = [first_word.text, second_word.text]
            result.append(element)

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='file name, should be .json', required=True)
    parser.add_argument('--bigrams', help='do you want to get bigrams', required=False, default=False)
    args = parser.parse_args()

    # extract
    preprocessor = DataPreprocessor(args.file)
    preprocessor.extract_data()
    preprocessor.print_extract_step_results()

    # transform
    preprocessor.transform_titles(args.bigrams)
    print('Completed cleaning titles')
    preprocessor.transform_abstracts(args.bigrams)
    print('Completed cleaning abstracts')
    preprocessor.transform_categories(args.bigrams)
    print('Completed cleaning categories')

    preprocessor.print_transform_step_results()
    output = preprocessor.combine_output()

    # load
    preprocessor.write_output('output', output)


if __name__:
    main()
