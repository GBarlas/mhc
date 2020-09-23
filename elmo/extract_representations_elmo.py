import csv
import fire
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from pathlib import Path


def elmo_features(dataset, output_path, tokens_path):
    tokens_path = Path(tokens_path)

    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    layers = ['default', 'word_emb', 'lstm_outputs1', 'lstm_outputs2', 'elmo']

    output_path = Path(output_path, 'elmo')
    for layer in layers:
        (output_path / layer).mkdir(parents=True)
    (output_path / 'tokens').mkdir(parents=True)

    with open(dataset, 'r', newline='') as fr:
        for values in csv.reader(fr):
            idx = values[0]
            if not idx.isdigit():
                continue
            with (tokens_path / ('%s.txt' % idx)).open('r') as fr:
                text = fr.read()
                tokens = text.split('\n')[3:]  # 3 tokens were added by ULMFiT
            (output_path / 'tokens' / idx).with_suffix(
                    '.txt').write_text('\n'.join(tokens))
            feats = elmo([' '.join(tokens)], signature="default", as_dict=True)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                features = sess.run(feats)
            for layer, feature in features.items():
                if layer == 'sequence_len':
                    continue
                np.save((output_path / layer / idx).with_suffix(
                    '.npy'), feature)


if __name__ == '__main__':
    fire.Fire(elmo_features)

