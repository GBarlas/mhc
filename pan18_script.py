import argparse
import csv
import json
from pathlib import Path


def from_path(path, counter2filename, predictions_path):
    for filename in [f for f in path.glob('*') if f.is_file()]:
        from_file(filename, counter2filename, predictions_path)

def from_file(filename, counter2filename, predictions_path):
    with filename.open() as fr:
        lines = [line.replace(' ', '').strip('\n') for line in fr.readlines()]
    predictions = []
    for line in lines:
        key, value = line.split('=')
        if key == 'Epoch':
            epoch = value
        elif key == 'NormalizationCorpus':
            C = value
        elif key == 'Counter':
            counter = value
        elif key == 'Rank':
            predicted_author = value.split(',')[0]
            predicted_author = str(int(predicted_author) + 1)
            predicted_author = "candidate000" + ('0' if len(predicted_author) == 1 else '') + predicted_author
            predictions.append({ "unknown-text":  counter2filename[counter], "predicted-author": predicted_author })
        elif key == 'End':
            (predictions_path / ('predictions_%s_%s' % (epoch, C))).write_text(json.dumps(predictions))
            predictions = []
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-f', '--file', 
            help="File exported from \"mhc.py\".")
    parser.add_argument(
            '-p', '--path', 
            help="""A path only with files exported from \"mhc.py\".""")
    parser.add_argument(
            '-d', '--dataset', 
            help="The \"csv\" created from \"create_dataset.py\" for PAN 2018 dataset", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    assert dataset.exists(), "File \"%s\" does not exists!" % dataset

    counter2filename = {}
    with open(dataset, newline='') as fr:
        for instance in csv.reader(fr):
            counter2filename[instance[0]] = instance[4]

    if args.path is not None:
        path = Path(args.path)
        assert path.exists(), "Path \"%s\" does not exists!" % path
        predictions_path = path / 'pan18-predictions'
        from_path(path, counter2filename, predictions_path)
    elif args.file is not None:
        filename = Path(args.file)
        assert filename.exists(), "File \"%s\" does not exists!" % filename
        predictions_path = filename.parent / 'pan18-predictions'
        from_file(filename, counter2filename, predictions_path)
    else:
        parser.print_help()

