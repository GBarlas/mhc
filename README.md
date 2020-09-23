# MHC

Code from the paper ["Cross-Domain Authorship Attribution Using Pre-trained Language Models"](https://link.springer.com/chapter/10.1007/978-3-030-49161-1_22).

## !!! Clone and setup 
This repository contains sub-modules, create a clone with `--recurse-submodules` option.

```
git clone --recurse-submodules https://github.com/GBarlas/mhc
```

Enter to `mhc` folder.
```
cd mhc
```

Install a conda environment for each pre-trained language model and MHC. 
All the environments require _gpu_ support, for _cpu_ support only, you may refer to: 
[BERT](https://github.com/google-research/bert), [ELMO](https://tfhub.dev/google/elmo/2), [GPT-2](https://github.com/openai/gpt-2), [ULMFiT](https://github.com/fastai/fastai) and [MHC](https://pytorch.org/). 

```
conda env create -f "envs/mhc-bert.yml"
conda env create -f "envs/mhc-elmo.yml"
conda env create -f "envs/mhc-gpt-2.yml"
conda env create -f "envs/mhc-ulmfit.yml"
conda env create -f "envs/mhc.yml"
```

Download the weights of the pre-trained language models.
```
sh download_weights.sh
```


## Usage

### Corpus to csv
You may use [CMCC](), [PAN18]() or your own corpus. The `corpus_to_csv.py` script will help you to create a compatible _csv_ file. To create the csv of CMCC or PAN18 just run the script with the required arguments. If you like to create a csv for another corpus you have to add a function into the `corpus_to_csv.py`, further information you may find into the script.

```
usage: corpus_to_csv.py [-h] {CMCC,PAN18} CORPUS-PATH CSV-FILENAME

positional arguments:
  {CMCC,PAN18}  Choose one.
  CORPUS-PATH   Where are the data?
  CSV-FILENAME  The filename of the output csv.

optional arguments:
  -h, --help    show this help message and exit
```

### Create an Authorship Attrubution problem
To create an AA problem, run the script `create_problem.py` with a selected corpus csv file and the UI will guide you step by step on how to define the training, evaluation and test set by selecting the approaprete instances from the corpus.

```
usage: create_problem.py [-h]
                         [--columns_and_desirables COLUMNS_AND_DESIRABLES]
                         corpus csv_filename

positional arguments:
  corpus                Choose a csv file created from "corpus_to_csv.py".
  csv_filename          The filename of the output csv.

optional arguments:
  -h, --help            show this help message and exit
  --columns_and_desirables COLUMNS_AND_DESIRABLES
                        This option must have a specific format. Run the
                        script once without this option and you will get it.
```

### Extract representations
Extract and save the representations that will be used as inputs to MHC. The representation from every layer of the selected language models will be saved along with the tokens, they have been created from.
!!! Don't forget to activate the coresponding conda environment `conda activate (mhc-bert|mhc-elmo|mhc-gpt-2|mhc-ulmfit)`

```
usage: extract_representations.py [-h] [--config_file CONFIG_FILE]
                                  [-e ELMO_TOKENS_PATH]
                                  {bert,elmo,ulmfit,gpt-2} dataset output_path

positional arguments:
  {bert,elmo,ulmfit,gpt-2}
                        Choose one.
  dataset               Select a dataset or a problem.
  output_path           Where would you like to save the representations?

optional arguments:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Choose a configuration file other than "config.ini"
  -e ELMO_TOKENS_PATH, --elmo_tokens_path ELMO_TOKENS_PATH
                        If "elmo" is selected, you need to define the tokens
                        folder path created from "ulmfit" from the same csv
                        file.
```

### Run MHC
`mhc.py` script will train and evaluate the MHC.
!!! Don't forget to activate the mhc conda environment `conda activate mhc`

```
usage: mhc.py [-h] -r REPRESENTATIONS_PATH -p PROBLEM -o OUTPUT_FILENAME
              [--epochs EPOCHS] [--load-model LOAD_MODEL]
              [--save-model-path SAVE_MODEL_PATH]
              [--vocabulary-size VOCABULARY_SIZE] [--device {cuda,cpu}]
              [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -r REPRESENTATIONS_PATH, --representations-path REPRESENTATIONS_PATH
                        The folder with the representations created by
                        "create_representations.py"
  -p PROBLEM, --problem PROBLEM
                        A "csv" file created by "create_problem.py"
  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
  --epochs EPOCHS
  --load-model LOAD_MODEL
                        Load model to continue training.
  --save-model-path SAVE_MODEL_PATH
                        Path to save the model.
  --vocabulary-size VOCABULARY_SIZE
  --device {cuda,cpu}
  --seed SEED
  ```

## Citation

Please use the following bibtex entry:
```
@inproceedings{barlas2020cross,
  title={Cross-Domain Authorship Attribution Using Pre-trained Language Models},
  author={Barlas, Georgios and Stamatatos, Efstathios},
  booktitle={IFIP International Conference on Artificial Intelligence Applications and Innovations},
  pages={255--266},
  year={2020},
  organization={Springer}
}
```
