import argparse
from pathlib import Path


def from_path(path):
    for filename in path.glob('*'):
        from_file(filename)


def from_file(filename):
    correct = 0
    n = 0
    with filename.open() as fr:
         lines = [line.replace(' ', '').strip('\n')                     
                 for line in fr.readlines()]
    for line in lines:
        key, value = line.split('=')
        if key == 'Epoch':
            epoch = value
        elif key == 'NormalizationCorpus':
            C = value
        elif key == 'Author':
            true_author = value
        elif key == 'Rank':
            winner_author = value.split(',')[0]
            if true_author == winner_author:
                correct += 1
            n += 1
        elif key == 'End':
            accuracy = str(correct / n)
            print(','.join([str(filename), epoch, C, accuracy]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-f', '--file', help="File exported from \"mhc.py\".")
    parser.add_argument(
            '-p', '--path', 
            help="""A path only with files exported from \"mhc.py\".""")
    args = parser.parse_args()

    if args.path is not None:
        path = Path(args.path)
        assert path.exists(), "Path \"%s\" does not exists!" % path
        from_path(path)
    elif args.file is not None:
        filepath = Path(args.file)
        assert filepath.exists(), "Path \"%s\" does not exists!" % filepath
        from_file(filepath)
    else:
        parser.print_help()
