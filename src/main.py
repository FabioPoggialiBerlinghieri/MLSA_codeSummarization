import argparse
import json
import os
import sys
from typing import Literal

import pandas as pd
import torch
import yaml
from datasets import load_dataset

import paddingHandler
from createDataset import code_padding_handler, text_padding_handler, code_vocabulary
from dataset import CodeDataset
from tokenizer import CodeTokenizer, EnglishTextTokenizer
from vocabulary_generator import VocabularyGenerator, EnglishVocabularyGenerator, PythonVocabularyGenerator

parser = argparse.ArgumentParser(description="TBD")

parser.add_argument('--config', type=str, required=True, help="YAML file path")

parser.add_argument('--max-code-len', type=int, default=None, help="Max code length")
parser.add_argument('--max-sum-len', type=int, default=None, help="Max text length")
parser.add_argument('--save-every', type=int, default=None, help="Save model weights every n epochs")
parser.add_argument('--resume', type=str, default=None, help="Resume file path")

args = parser.parse_args()

config_filepath = args.config
max_code_len = args.max_code_len
max_text_len = args.max_sum_len
save_every = args.save_every
resume = args.resume

class VocabularyStoreHandler:

    @staticmethod
    def store_vocabularies(vocabulary_generator, voc_path):
        vocabulary = vocabulary_generator.generate()
        with open(voc_path, "w") as f:
            json.dump(vocabulary, f, indent=4)

        return vocabulary

class CodeDatasetHandler:

    def __init__(self, config_filepath, max_code_len, max_sum_len):

        self.config_filepath = config_filepath
        try:
            with open(self.config_filepath, 'r') as file:
                self.config_yaml = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Fatal error: '{config_filepath}' does not exist!")
            sys.exit(1)

        self.max_code_len = max_code_len if max_code_len is not None else self.config_yaml['max_code_len']
        self.max_sum_len = max_sum_len if max_sum_len is not None else self.config_yaml['max_sum_len']
        self.modified = self.max_code_len != self.config_yaml['max_code_len'] or self.max_sum_len != self.config_yaml['max_sum_len']

        self.dataset_link = self.config_yaml['dataset_link']

        self.dataset_train_path = self.config_yaml['dataset_train_path']
        self.dataset_val_path = self.config_yaml['dataset_val_path']
        self.dataset_test_path = self.config_yaml['dataset_test_path']

        self.python_voc_path = self.config_yaml['python_voc_path']
        self.english_voc_path = self.config_yaml['english_voc_path']

        if self.config_yaml['first_time'] is True:
            self.modified = True
            self.config_yaml['first_time'] = False
            self.__save_config()

        self.codeTokenizer = None
        self.text_padding_handler = None
        self.englishTokenizer = None
        self.code_padding_handler = None

    def load_dataset(self, split: Literal['train', 'val', 'test']):
        if self.modified:
            self.config_yaml['max_code_len'] = self.max_code_len
            self.config_yaml['max_sum_len'] = self.max_sum_len
            self.__save_config()
            self.create_dataset()
        if split == 'train':
            path = self.dataset_train_path
        elif split == 'val':
            path = self.dataset_val_path
        else:
            path = self.dataset_test_path
        dataframe = pd.read_json(path, orient='records')
        return CodeDataset(torch.tensor(dataframe["code"], dtype=torch.long),
                           torch.tensor(dataframe["text"], dtype=torch.long))

    def __save_config(self):
        with open(self.config_filepath, 'w') as file:
            yaml.dump(self.config_yaml, file, default_flow_style=False, sort_keys=False)

    def __tokenize_pad(self, code: str, text: str) -> tuple[list[int], list[int]]:

        return (code_padding_handler.padding(self.codeTokenizer.tokenize(code)),
                text_padding_handler.padding(self.englishTokenizer.tokenize('[CLS]') + self.englishTokenizer.tokenize(text)
                                             + self.englishTokenizer.tokenize('[SEP]')))

    def create_dataset(self):

        # create vocabulary
        dataset = load_dataset(self.dataset_link, split="train").to_pandas()

        snippets = dataset["code"]
        texts = dataset["text"]
        code_vocabulary_generator = PythonVocabularyGenerator(snippets)
        text_vocabulary_generator = EnglishVocabularyGenerator(texts)

        dict_code_vocabulary = VocabularyStoreHandler.store_vocabularies(code_vocabulary_generator, self.python_voc_path)
        dict_text_vocabulary = VocabularyStoreHandler.store_vocabularies(text_vocabulary_generator, self.english_voc_path)

        main_keywords = code_vocabulary_generator.get_main_keywords()
        self.codeTokenizer = CodeTokenizer(dict_code_vocabulary, main_keywords)
        self.code_padding_handler = paddingHandler.PaddingHandler(self.max_code_len)

        self.englishTokenizer = EnglishTextTokenizer(dict_text_vocabulary)
        self.text_padding_handler = paddingHandler.PaddingHandler(1 + self.max_sum_len) #CLS + TEXT LEN

        # then create dataset
        splits_map = {
            self.dataset_train_path: "train",
            self.dataset_val_path: "validation",
            self.dataset_test_path: "test"
        }

        for path, target in splits_map.items():
            dataset = load_dataset(self.dataset_link, split=target).to_pandas()

            tok_dataset = dataset.apply(
                lambda row: self.__tokenize_pad(row["code"], row["text"]),
                axis=1,
                result_type="expand"
            )
            tok_dataset.columns = ["code", "text"]

            tok_dataset.to_json(path, orient="records", indent=2)












