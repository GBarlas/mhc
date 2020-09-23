import argparse
import csv
import subprocess
import configparser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # to run on CPU


"""Extract the representations for each language model for a seleced corpus or
problem. The desirable configurations for each language model are define from
the selected configuration "ini" file.

"""


METHODS = ['bert', 'elmo', 'ulmfit', 'gpt-2']
config = configparser.ConfigParser()


def extract_representations(args):
    config.read(args.config_file)
    from_config = []
    for name, value in config.items(args.method):
        from_config.append(('--%s=%s' % (name, value)).replace('=True', ''))
    cmd = []
    cmd.append('python')
    cmd.append(os.path.join(
        args.method, 'extract_representations_%s.py' % args.method))
    cmd.append('--dataset=%s' % args.dataset)
    cmd.append('--output_path=%s' % args.output_path)
    cmd.extend(from_config)
    if args.method == 'elmo':
        cmd.append('--tokens_path=%s' % args.elmo_tokens_path)
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'method', choices=METHODS,
            help="Choose one.")
    parser.add_argument(
            'dataset',
            help="Select a dataset or a problem.")
    parser.add_argument(
            'output_path',
            help="Where would you like to save the representations?")
    parser.add_argument(
            '--config_file', default='config.ini',
            help="Choose a configuration file other than \"config.ini\"")
    parser.add_argument(
            '-e', '--elmo_tokens_path',
            help="""If \"elmo\" is selected, you need to define the tokens
                 folder path created from \"ulmfit\" from the same csv
                 file.""")
    args = parser.parse_args()

    assert args.method == 'elmo' and args.elmo_tokens_path is not None, \
            "ELMo requires '--elmo_tokens_path' option"

    extract_representations(args)


if __name__ == '__main__':
    main()

