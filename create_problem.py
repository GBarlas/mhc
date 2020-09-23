# -*- coding: utf-8 -*-
import argparse
import csv
from pathlib import Path


"""Create a csv file with the same columns as the given csv. The given
csv (corpus) containes all instances of the corpus and with this script
it's easy to create an authorship attribution problem. The values of the
instances are remaining the same except from column 2 which indicates
the training, evaluation or test set.

"""


SETS = {'TRAINING': 'train',
        'EVALUATION': 'eval',
        'TEST': 'test'
        }


def get_desirable_values(corpus):
    with corpus.open('r', newline='') as fr:
        reader = csv.reader(fr)
        # Choose columns
        header = next(reader)
        print('')
        print("Choose colunm numbers seperated by coma:")
        print('\n'.join(["   %d - %s" % (i, h) for i, h in enumerate(header)]))
        print('')
        columns = [int(c) for c in input('>> ').split(',')]
        n_columns = len(columns)

        # Get unique values per column
        data = [[] for _ in range(n_columns)]
        for values in csv.reader(fr):
            [data[i].append(values[c]) for i, c in enumerate(columns)]
        data = [sorted(set(data[i])) for i in range(n_columns)]

        # Choose the desirable values per set (training, evaluation, test)
        desirables = {}
        for SET in SETS.keys():
            desirables[SET] = []
            for i, c in enumerate(columns):
                print('')
                print("Choose the desirable values for the %s set:" % SET)
                print("   Header           -  %s" % header[c])
                str_data = list(map(str, data[i]))
                if ''.join(str_data).isdigit():
                    print("   Available values - {%s}" % ', '.join(
                        sorted(str_data, key=int)))
                else:
                    print("   Available values - {%s}" % ', '.join(
                        str_data))
                print('')
                desirables[SET].append(input('>> ').split(','))

        return columns, desirables


def create_csv(corpus, output, columns, desirables):
    output.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open('r', newline='') as fr, \
    output.open('w', newline='') as fw:
        reader = csv.reader(fr)
        writer = csv.writer(fw)
        writer.writerow(next(reader))
        for instance in reader:
            set_type = '<UNK>'
            for SET in SETS.keys():
                if all(instance[c] in desirables[SET][i]
                        for i, c in enumerate(columns)):
                    set_type = SETS[SET]
            if set_type == '<UNK>':
                continue
            instance[1] = set_type
            writer.writerow(instance)


def get_header(corpus):
    with corpus.open('r', newline='') as fr:
        return next(csv.reader(fr))


def create_problem(corpus, output, columns_and_desirables=None):
    if columns_and_desirables is None:
        columns, desirables = get_desirable_values(corpus)
    else:
    	columns, desirables = columns_and_desirables

    create_csv(corpus, output, columns, desirables)

    header = get_header(corpus)

    print('')
    print("Created '%s' from '%s'." % (output, corpus))
    print("Selected columns:")
    for c in columns:
        print("   %d - %s" % (c, header[c]))

    print('')
    print("Selected values for each corpus:")
    for k, v in desirables.items():
        print("%s" % k)
        for i, val in enumerate(v):
            print("   %d - %s" % (columns[i], ','.join(str(s) for s in val)))
        print('')
    print("For future use:")
    print('--columns_and_desirables="', (columns, desirables), '"')
    print('Done!')
    print('')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'corpus',
            help="Choose a csv file created from \"corpus_to_csv.py\".")
    parser.add_argument(
            'csv_filename',
            help="The filename of the output csv.")
    parser.add_argument(
            '--columns_and_desirables',
            help="""This option must have a specific format. Run the script once
                    without this option and you will get it.""")
    args = parser.parse_args()

    corpus = Path(args.corpus).resolve()
    output = Path(args.output).resolve()

    create_problem(corpus, output, eval(str(args.columns_and_desirables)))


if __name__ == '__main__':
    main()

