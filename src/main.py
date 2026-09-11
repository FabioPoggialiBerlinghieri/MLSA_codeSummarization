import json
import os
from typing import Literal
import pandas as pd
import torch
import yaml
from datasets import load_dataset
import paddingHandler
from dataset import CodeDataset
from tokenizer import CodeTokenizer, EnglishTextTokenizer
from vocabulary_generator import EnglishVocabularyGenerator, PythonVocabularyGenerator

class VocabularyStoreHandler:

    @staticmethod
    def store_vocabularies(vocabulary_generator, voc_path):
        vocabulary = vocabulary_generator.generate()
        with open(voc_path, "w") as f:
            json.dump(vocabulary, f, indent=4)

        return vocabulary

    @staticmethod
    def load_vocabulary(voc_path):
        if not os.path.exists(voc_path):
            raise FileNotFoundError(f"Can't find vocabulary file: {voc_path}")

        with open(voc_path, "r", encoding="utf-8") as f:
            return json.load(f)

class DatasetHandler:

    def __init__(self, config_yaml, max_code_len, max_sum_len):
        self.config_yaml = config_yaml
        self.max_code_len = max_code_len
        self.max_sum_len = max_sum_len

        self.dataset_link = self.config_yaml['dataset_link']
        self.dataset_config = self.config_yaml['dataset_config']

        self.dataset_train_path = self.config_yaml['dataset_train_path']
        self.dataset_val_path = self.config_yaml['dataset_val_path']
        self.dataset_test_path = self.config_yaml['dataset_test_path']

        self.python_voc_path = self.config_yaml['python_voc_path']
        self.english_voc_path = self.config_yaml['english_voc_path']

        self.modified = False
        if self.config_yaml['first_time'] is True:
            self.modified = True
            self.config_yaml['first_time'] = False

        self.codeTokenizer = None
        self.text_padding_handler = None
        self.englishTokenizer = None
        self.code_padding_handler = None

        try:
            with open("../data/main_keywords.json", "r", encoding="utf-8") as file:
                self.main_keywords = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            self.main_keywords = None

        self.yaml_path = None

    def load_dataset(self, split: Literal['train', 'val', 'test']):
        if self.modified:
            self.__save_config()
            self.create_dataset()
        else:
            self.__load_vocabularies()

        if split == 'train':
            path = self.dataset_train_path
        elif split == 'val':
            path = self.dataset_val_path
        else:
            path = self.dataset_test_path
        dataframe = pd.read_json(path, orient='records')
        return CodeDataset(torch.tensor(dataframe["code"], dtype=torch.long),
                           torch.tensor(dataframe["text"], dtype=torch.long))

    def set_yaml_path(self, yaml_path):
        self.yaml_path = yaml_path

    def __save_config(self):
        if self.yaml_path is None:
            raise FileNotFoundError(f"Can't find yaml file: {self.yaml_path}")

        with open(self.yaml_path, 'w') as file:
            yaml.dump(self.config_yaml, file, default_flow_style=False, sort_keys=False)

    def __tokenize_pad(self, code: str, text: str) -> tuple[list[int], list[int]]:

        return (self.code_padding_handler.padding(self.codeTokenizer.tokenize(code)),
                self.text_padding_handler.padding(self.englishTokenizer.tokenize('[CLS]') + self.englishTokenizer.tokenize(text)
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
        self.main_keywords = main_keywords

        with open("../data/main_keywords.json", "w", encoding="utf-8") as file:
            json.dump(self.main_keywords, file, indent=4, ensure_ascii=False)

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
            dataset = load_dataset(self.dataset_link, self.dataset_config, split=target).to_pandas()

            tok_dataset = dataset.apply(
                lambda row: self.__tokenize_pad(row["code"], row["docstring"]),
                axis=1,
                result_type="expand"
            )
            tok_dataset.columns = ["code", "docstring"]

            tok_dataset.to_json(path, orient="records", indent=2)

    def __load_vocabularies(self):

        dict_code_vocabulary = VocabularyStoreHandler.load_vocabulary(self.python_voc_path)
        dict_text_vocabulary = VocabularyStoreHandler.load_vocabulary(self.english_voc_path)

        self.codeTokenizer = CodeTokenizer(dict_code_vocabulary, self.main_keywords)
        self.code_padding_handler = paddingHandler.PaddingHandler(self.max_code_len)

        self.englishTokenizer = EnglishTextTokenizer(dict_text_vocabulary)
        self.text_padding_handler = paddingHandler.PaddingHandler(1 + self.max_sum_len)  # CLS + TEXT LEN
