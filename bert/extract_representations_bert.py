# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import csv
import sys
from pathlib import Path
import numpy as np

from bert import modeling
from bert import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None, "")

flags.DEFINE_string("output_path", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter"
        "than this will be padded.")

flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
        "vocab_file", None,
        "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_string(
        "master", None, "If using a TPU, the address of the master.")

flags.DEFINE_bool(
        "use_one_hot_embeddings", False,
        "If True, tf.one_hot will be used for embedding lookups, otherwise "
        "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
        "since it is much faster.")


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
                "unique_ids": tf.constant(
                    all_unique_ids,
                    shape=[num_examples], dtype=tf.int32),
                "input_ids": tf.constant(
                    all_input_ids,
                    shape=[num_examples, seq_length], dtype=tf.int32),
                "input_mask": tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length], dtype=tf.int32),
                "input_type_ids": tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length], dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(
        bert_config, init_checkpoint, layer_indexes, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
        all_layers = model.get_all_encoder_layers()
        predictions = {
                "unique_id": unique_ids,
        }
        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def convert_text_to_features(text, seq_length, tokenizer):

    toks_seq_length = seq_length - 2  # Account for [CLS] and [SEP] with "- 2"
    input_mask = [1] * seq_length
    input_type_ids = [0] * seq_length

    features = []
    unique_id = 0
    text = tokenization.convert_to_unicode(text)
    toks = tokenizer.tokenize(text)
    for i in range(0, len(toks), toks_seq_length):

        tokens = ["[CLS]"] + toks[i:i + toks_seq_length] + ["[SEP]"]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        while len(input_ids) < seq_length:
            input_ids.append(0)

        assert len(input_ids) == seq_length, "%d, %d" % (
                len(input_ids), seq_length)
        assert len(input_mask) == seq_length, "%d, %d" % (
                len(input_ids), seq_length)
        assert len(input_type_ids) == seq_length, "%d, %d" % (
                len(input_type_ids), seq_length)

        features.append(
            InputFeatures(
                unique_id=unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
        unique_id += 1
    return features


def main(_):
    layer_indexes = [x for x in FLAGS.layers.split(",")]

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
            master=FLAGS.master,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    num_shards=None,
                    per_host_input_for_training=is_per_host))

    output_path = Path(FLAGS.output_path, 'bert')
    for l in layer_indexes:
        (output_path / l).mkdir(parents=True, exist_ok=True)
    (output_path / 'tokens').mkdir(parents=True, exist_ok=True)

    with open(FLAGS.dataset, 'r', newline='') as fr:
        for values in csv.reader(fr):
            idx = values[0]
            if not idx.isdigit():
                continue
            features = convert_text_to_features(
                    text=values[3],
                    seq_length=FLAGS.max_seq_length,
                    tokenizer=tokenizer)

            unique_id_to_feature = {}
            for feature in features:
                unique_id_to_feature[feature.unique_id] = feature

            model_fn = model_fn_builder(
                    bert_config=bert_config,
                    init_checkpoint=FLAGS.init_checkpoint,
                    layer_indexes=map(int, layer_indexes),
                    use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

            # If TPU is not available, this will fall back to normal Estimator
            # on CPU or GPU.
            estimator = tf.contrib.tpu.TPUEstimator(
                    use_tpu=False,
                    model_fn=model_fn,
                    config=run_config,
                    predict_batch_size=FLAGS.batch_size)

            input_fn = input_fn_builder(
                    features=features, seq_length=FLAGS.max_seq_length)

            tokens = []
            arrays = dict(zip(
                layer_indexes,
                [np.array([]) for _ in range(len(layer_indexes))]))
            arrays = {}
            for result in estimator.predict(
                    input_fn, yield_single_examples=True):
                toks = unique_id_to_feature[result["unique_id"]].tokens
                tokens.extend(toks)
                for (j, layer_index) in enumerate(layer_indexes):
                    rep = result["layer_output_%d" % j][:len(toks)]
                    if layer_index not in arrays:
                        arrays[layer_index] = rep
                    else:
                        arrays[layer_index] = np.concatenate(
                                (arrays[layer_index], rep))

            (output_path / 'tokens' / idx).with_suffix(
                    '.txt').write_text('\n'.join(tokens))
            for (j, layer_index) in enumerate(layer_indexes):
                np_file = (output_path / layer_index / idx).with_suffix('.npy')
                np.save(np_file, arrays[layer_index])


if __name__ == "__main__":
    flags.mark_flag_as_required("dataset")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_path")
    tf.app.run()

