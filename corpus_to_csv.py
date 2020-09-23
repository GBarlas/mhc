# -*- coding: utf-8 -*-
import argparse
import json
import csv
import random
from pathlib import Path


"""Create a csv file
col1: (int) counter
col2: {train, eval, test, <UNK>}
      for {training set, evaluation set, test set, unknown} respectively
col3: (int) author id
col4: (str) text
the rest columns depend on corpus.

Currently, the supported corpuses may seen below after the comments in CORPUS
list.

"""

CORPUS = {
        'CMCC': ['counter',
                 '<UNK>',
                 'author_id',
                 'text',
                 'filename',
                 'author',
                 'genre',
                 'genre_number',
                 'topic',
                 'topic_number'],
        'PAN18': ['counter',
                  'set_type',
                  'candidate_id',
                  'text',
                  'filename',
                  'file_number',
                  'problem',
                  'problem_number',
                  'candidate',
                  'language']
        }


# Add a new corpus template
# !!! Don't forget to add your corpus to CORPUS dictionary above.
def corpus_name(corpus_path):
    # In each call the function must return one instance of the corpus as list.
    i = "(int) counter"
    set_type = """{train, eval, test, <UNK>} for
                {training set, evaluation set, test set, unknown}
                respectively"""
    author_id = "(int) author id"
    text = "(str) text"
    other = """the rest of the list may containe any other
            "usefull information from the corpus"""

    yield [i, set_type, author_id, text, author, other]


def cmcc(corpus_path):
    i = 0
    for txtfile in corpus_path.glob('**/*.txt'):
        with txtfile.open(encoding='ISO-8859-1') as fr:
            filename = txtfile.name
            i += 1
            author = filename[:3] if filename[2].isdigit() else filename[:2]
            ret = []
            ret.append(i)
            ret.append('<UNK>')  # set_type
            ret.append(author[1:])  # author_id
            ret.append(fr.read())  # text
            ret.append(filename)
            ret.append(author)  # author
            ret.append(filename[-8])  # genre
            ret.append(filename[-7])  # genre_number
            ret.append(filename[-6])  # topic
            ret.append(filename[-5])  # topic_number
            yield ret


def pan18(corpus_path):
    collection_info = json.load((corpus_path / 'collection-info.json').open())
    i = 0
    for problem in collection_info:
        if problem['language'] != 'en':
            continue
        problem_folder = corpus_path / problem['problem-name']
        problem_info = json.load((problem_folder / 'problem-info.json').open())
        for candidate in problem_info["candidate-authors"]:
            candidate_folder = problem_folder / candidate["author-name"]
            for f in candidate_folder.iterdir():
                filename = f.name
                i += 1
                candidate_id = int(
                        candidate["author-name"].replace('candidate', ''))
                text = (candidate_folder / filename).read_text(
                        encoding=problem['encoding'])
                problem_number = int(
                        problem["problem-name"].replace('problem', ''))
                file_number = int(filename[5:10])
                ret = []
                ret.append(i)
                ret.append('train')
                ret.append(candidate_id)
                ret.append(text)
                ret.append(filename)
                ret.append(file_number)
                ret.append(problem["problem-name"])
                ret.append(problem_number)
                ret.append(candidate["author-name"])
                ret.append(problem["language"])
                yield ret

        ground_truth = json.load(
                (problem_folder / 'ground-truth.json').open())["ground_truth"]
        unknown_folder = problem_folder / problem_info["unknown-folder"]
        for unknown in ground_truth:
            i += 1
            if unknown["true-author"] == "<UNK>":
                candidate_id = -1
            else:
                candidate_id = int(
                        unknown["true-author"].replace('candidate', ''))
            text = (unknown_folder / unknown["unknown-text"]).read_text(
                    encoding=problem['encoding'])
            problem_number = int(
                    problem["problem-name"].replace('problem', ''))
            file_number = int(
                    unknown["unknown-text"][7:12])
            ret = []
            ret.append(i)
            ret.append('test')
            ret.append(candidate_id)
            ret.append(text)
            ret.append(unknown["unknown-text"])
            ret.append(file_number)
            ret.append(problem["problem-name"])
            ret.append(problem_number)
            ret.append(unknown["true-author"])
            ret.append(problem["language"])
            yield ret


def create_csv(corpus, corpus_path, output):
    with output.open('w', newline='') as fw:
        writer = csv.writer(fw)
        writer.writerow(CORPUS[corpus])
        for instance in get_generator(corpus, corpus_path):
            writer.writerow(instance)

    print('')
    print("Created '%s' from '%s' folder." % (output, corpus_path))
    print("Header: '%s'." % ','.join(CORPUS[corpus]))
    print('Done!')
    print('')


def get_generator(corpus, corpus_path):
    fun_name = corpus.lower().replace('-', '_')
    generator = globals()[fun_name](corpus_path)
    return generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', choices=CORPUS.keys(), help="Choose one.")
    parser.add_argument('corpus_path', help="Where are the data?")
    parser.add_argument('csv_filename', help="The filename of the output csv.")
    args = parser.parse_args()

    args.corpus_path = Path(args.corpus_path).resolve()
    args.csv_filename = Path(args.csv_filename).with_suffix('.csv')

    create_csv(args.corpus, args.corpus_path, args.csv_filename)


if __name__ == '__main__':
    main()

