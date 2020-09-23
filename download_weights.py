#!/bin/bash

echo "Downloading weights for BERT"
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -P bert
unzip bert/uncased_L-12_H-768_A-12.zip -d bert/uncased_L-12_H-768_A-12

echo "Downloading weights for ULMFiT"
wget http://files.fast.ai/models/wt103/fwd_wt103.h5 -P ulmfit/wt103
wget http://files.fast.ai/models/wt103/fwd_wt103_enc.h5 -P ulmfit/wt103
wget http://files.fast.ai/models/wt103/itos_wt103.pkl -P ulmfit/wt103

echo "Downloading weights for GPT-2"
cd gpt-2/gpt-2
python download_model.py 124M
cd ../..

