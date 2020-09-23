import argparse
import csv
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader


def get_sets(problem, author2id):
    # get train/eval/test 
    with problem.open('r', newline='') as fr:
        train_set = {}
        eval_set = {}
        test_set = {}
        for counter, set_, author, *_  in csv.reader(fr):
            if set_ == 'train':
                train_set[counter] = int(author2id[author])
            elif set_ == 'eval':
                eval_set[counter] = int(author2id[author])
            elif set_ == 'test':
                test_set[counter] = int(author2id[author])
    return train_set, eval_set, test_set


class MHC(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear_io = torch.nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=True)

    def forward(self, input):
        return self.linear_io(input)


def get_representations(representations_path, set_):
    for counter, author_id in set_.items():
        yield (counter,
               author_id, 
               (representations_path / '..' / 'tokens' / counter).with_suffix('.txt').read_text().split('\n'),
               np.load((representations_path / counter).with_suffix('.npy')).squeeze())

def main(args):
    device = torch.device(args.device)
    
    # define seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Paths
    representations_path = Path(args.representations_path)
    assert representations_path.exists()
    problem_path = Path(args.problem)
    assert problem_path.exists()
    
    # author2id dict
    author_ids = sorted(set([n[2] for n in csv.reader(problem_path.open('r', newline='')) if n[2].isdigit()]), key=int)
    n_authors = len(author_ids)
    author2id = dict(zip(author_ids, range(n_authors)))
    
    # token2id dict
    tokens_freq = {}
    for token_file in (representations_path /  '..'/ 'tokens').glob('*.txt'):
        with token_file.open('r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip()
                tokens_freq[line] = tokens_freq.setdefault(line, 0) + 1

    tmp_dict = {}
    for i, (k, v) in enumerate(sorted(tokens_freq.items(), key=lambda t: t[1], reverse=True)):
        if i < args.vocabulary_size:
            tmp_dict[k] = v
    
    tokens = tmp_dict.keys()
    tokens_tensor = torch.LongTensor(list([i] for i in range(len(tokens)))).to(device)
    token2id = dict(zip(tokens, [i for i in range(len(tokens))]))
    
    random_npy = next(representations_path.glob('*.npy'))
    input_size = np.load(random_npy).squeeze().shape[-1] 
    vocabulary_size = len(token2id)
    output_size = vocabulary_size

    results_file = Path(args.output_filename)
    if results_file.exists():
        print("File \"%s\" already exists! Results will be appended to the existing file." % results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    save_model_path = Path(args.save_model_path, results_file.stem)
    save_model_path.mkdir(parents=True, exist_ok=True)

    model = []
    optimizer = []
    for i_author in range(n_authors):
        epoch = 0
        model.append(MHC(input_size, output_size))
        model[-1].to(device)
        optimizer.append(torch.optim.Adagrad(model[-1].parameters())) 
        
    # load model
    if args.load_model:
        for i_author in range(n_authors):
            state = torch.load(args.load_model % i_author)
            epoch = state['epoch'] + 1
            model[i_author].load_state_dict(state['state_dict'])
            optimizer[i_author].load_state_dict(state['optimizer'])


    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    train_set, eval_set, test_set = get_sets(problem_path, author2id)
    train_data = [(counter, author_id, tokens, torch.Tensor(representations)) for counter, author_id, tokens, representations in get_representations(representations_path, train_set)]
    eval_data = [(counter, author_id, tokens, torch.Tensor(representations)) for counter, author_id, tokens, representations in get_representations(representations_path, eval_set)]
    test_data = [(counter, author_id, tokens, torch.Tensor(representations)) for counter, author_id, tokens, representations in get_representations(representations_path, test_set)]


    ones = torch.ones(n_authors).long().to(device)
    n = torch.LongTensor(0).to(device)
    for epoch in range(epoch, args.epochs):
        print('Epoch: %d' % epoch)
        
        if len(eval_data) == 0: eval_data = train_data
        n_vectors = {}
        for set_type, data in zip(['train', 'eval', 'test'], [train_data, eval_data, test_data]):

            training = set_type == 'train'
            if training: 
                for i_author in range(n_authors):
                    model[i_author].train()
            else: 
                for i_author in range(n_authors):
                    model[i_author].eval()
            
            counters = []
            author_ids = []
            loss_per_author = torch.zeros(n_authors, len(data)).to(device)
            i2counter = {}
            for i, (counter, author_id, tokens, representations) in enumerate(tqdm(data, desc='%s set' % set_type, total=len(data))):
                i2counter[i] = counter

                representations = representations.to(device)
                author_ids.append(author_id)
                counters.append(counter)
                temp_loss_per_author = torch.zeros(n_authors).to(device)
                n = 0
                for token, feature in zip(tokens[1:], representations):
                    if token not in token2id: continue
                    n += 1

                    if training:
                        output = model[author_id].forward(feature)
                        En = torch.nn.functional.cross_entropy(output.reshape(1, vocabulary_size), tokens_tensor[token2id[token]], reduction='none')
                        temp_loss_per_author[author_id] += En.data[0]

                        En.backward()
                        optimizer[author_id].step()
                        model[author_id].zero_grad()
                        continue
                    else:
                        for i_author in range(n_authors):
                            output = model[i_author].forward(feature)
                            En = torch.nn.functional.cross_entropy(output.reshape(1, vocabulary_size), tokens_tensor[token2id[token]], reduction='none')
                            temp_loss_per_author[i_author] += En.data[0]

                loss_per_author[:, i] += (temp_loss_per_author / n)

            if set_type == 'train': 
                n_vectors['None'] = torch.zeros_like(loss_per_author.mean(1)) 
                continue
            elif set_type == 'eval':
                n_vectors['K' if eval_data == train_data else 'C'] = loss_per_author.mean(1)
                continue
            elif set_type == 'test':
                n_vectors['U'] = loss_per_author.mean(1)

            # save results
            with results_file.open('a') as fa:
                print("Epoch=%d" % epoch, file=fa)
                for C, n_vector in n_vectors.items():
                    print("  NormalizationCorpus=%s" % C, file=fa), 
                    print("  NormalizationVector=%s" % ','.join([str(float(x)) for x in n_vector]), file=fa)
                    for n, a, l in zip(counters, author_ids, (loss_per_author.t() - n_vector)):
                        print("    Counter=%s" % n, file=fa)
                        print("    Author=%d" % a, file=fa)
                        e, r = l.sort(descending=False)
                        print("    Rank=%s" % ','.join([str(int(x)) for x in r]), file=fa)
                        print("    NormalizedError=%s" % ','.join([str(float(x)) for x in e]), file=fa)
                    print("  End=%s" % C, file=fa)
        
        # save model
        for i in range(n_authors):
            state = {
                'epoch': epoch,
                'state_dict': model[i].state_dict(),
                'optimizer': optimizer[i].state_dict(),
                }
            torch.save(state,  save_model_path / ('model_%d_%d.model' % (i, epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--representations-path', 
            help="The folder with the representations created by \"create_representations.py\"", required=True)
    parser.add_argument('-p', '--problem', help="A \"csv\" file created by \"create_problem.py\"", required=True)
    parser.add_argument('-o', '--output-filename', required=True)
    parser.add_argument('--epochs', type=int, default=21)
    parser.add_argument('--load-model', help="Load model to continue training.")
    parser.add_argument('--save-model-path', help="Path to save the model.", default='saved_models')
    parser.add_argument('--vocabulary-size', type=float, default=1000)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--seed', type=int, default=5)
    args = parser.parse_args()

    main(args)

