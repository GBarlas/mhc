import csv
import fire
import pickle
import sys
from tqdm import tqdm
from pathlib import Path
sys.path.append('ulmfit/fastai/old')
from fastai.text import *


def get_model(itos_filename, lm_filename):
    """Load the classifier and int to string mapping

    Args:
        itos_filename (str): The filename of the int to string mapping file
        (usually called itos.pkl)
        lm_filename (str): The filename of the trained language model

    Returns:
        string to int mapping, model
    """

    # load the int to string mapping file
    itos = pickle.load(Path(itos_filename).open('rb'))
    # turn it into a string to int mapping (which is what we need)
    stoi = collections.defaultdict(
            lambda: 0, {str(v): int(k) for k, v in enumerate(itos)})

    # these parameters aren't used, but this is the easiest way to get a model
    bptt, em_sz, nh, nl = 70, 400, 1150, 3
    vs = len(itos)

    model = get_rnn_classifer(
            bptt, 20000*70, 21, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl,
            pad_token=1, layers=[em_sz*3, 50, 21], drops=[0., 0.])

    models = TextModel(model)
    load_model(models.model[0], lm_filename)
    model = models.model

    model.reset()
    model.eval()

    return stoi, model


def create_representations(stoi, model, text):
    """Do the actual prediction on the text using the
        model and mapping files passed
    """

    # prefix text with tokens:
    #   xbos: beginning of sentence
    #   xfld 1: we are using a single field here
    input_str = 'xbos xfld 1 ' + text

    # predictions are done on arrays of input.
    # We only have a single input, so turn it into a 1x1 array
    texts = [input_str]

    # tokenize using the fastai wrapper around spacy
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))

    # turn into integers for each word
    encoded = [stoi[p] for p in tok[0]]

    # we want a [x,1] array where x is the number
    #  of words inputted (including the prefix tokens)
    ary = np.reshape(np.array(encoded), (-1, 1))

    # turn this array into a tensor
    tensor = torch.from_numpy(ary)

    # wrap in a torch Variable
    variable = Variable(tensor)

    # do the predictions
    predictions = model(variable)

    return tok[0]


def save_representations(itos_filename, lm_filename, dataset, output_path):
    """
    Loads a model and produces predictions on arbitrary input.
    :param itos_filename: the path to the id-to-string mapping file
    :param trained_lm_filename: the filename of the trained language model;
                                        typically ends with "lm_1.h5"
    """

    # create output directories
    output_path = Path(output_path, 'ulmfit')
    for i in range(3):
        (output_path / str(i)).mkdir(parents=True)
    (output_path / 'tokens').mkdir(parents=True)

    # get model
    stoi, model = get_model(itos_filename, lm_filename)

    # extract representations and save them
    with open(dataset, 'r', newline='') as fr:
        for values in csv.reader(fr):
            idx = values[0]
            if idx.isdigit():
                text = values[3].strip().replace('\n', '')

                tok = create_representations(stoi, model, text)
                with (output_path / 'tokens' / idx).with_suffix(
                        '.txt').open('w') as fr:
                    fr.write('\n'.join(tok))
                [Path('ulmfit/tmp/ulmfit_representations-%s.npy' % i).rename((
                    output_path / i / idx).with_suffix(
                        '.npy')) for i in map(str, range(3))]


if __name__ == '__main__':
    fire.Fire(save_representations)

