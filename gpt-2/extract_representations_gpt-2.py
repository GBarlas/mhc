import fire
import json
import os
import csv
from pathlib import Path
from time import sleep
import numpy as np
import tensorflow as tf
import sys
sys.path.append('gpt-2/gpt-2/src')
import model, sample, encoder


def interact_model(
    dataset,
    output_path,
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir=None
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must
     divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As
     the temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """

    length = None if length == 'None' else length
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        output_path = Path(output_path, 'gpt-2')
        for l in map(str, range(12)):
            (output_path / l).mkdir(parents=True)
        (output_path / 'tokens').mkdir(parents=True)

        with open(dataset, 'r', newline='') as fr:
            for values in csv.reader(fr):
                idx = values[0]
                if not idx.isdigit():
                    continue
                text = values[3]
                context_tokens = enc.encode(text)
                (output_path / 'tokens' / idx).with_suffix(
                        '.txt').write_text('\n'.join(
                            str(ct) for ct in context_tokens))
                hiddens = np.zeros((
                    batch_size,
                    hparams.n_layer,
                    len(context_tokens),
                    hparams.n_embd))
                for n in range(0, len(context_tokens), length):
                    h = sess.run(
                            output,
                            feed_dict={
                                context:
                                [context_tokens[n:n+length]
                                    for _ in range(batch_size)]})
                    hiddens[:, :, n:n+length, :] = h

                for l in range(12):
                    np.save((output_path / str(l) / idx).with_suffix(
                        '.npy'), hiddens[0, l, :len(context_tokens), :])


if __name__ == '__main__':
    fire.Fire(interact_model)

