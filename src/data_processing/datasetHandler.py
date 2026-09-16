import json
import os
from typing import Literal
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from . import paddingHandler
from .dataset import CodeDataset
from .tokenizer import CodeTokenizer, EnglishTextTokenizer
from .vocabulary_generator import EnglishVocabularyGenerator, PythonVocabularyGenerator
from tqdm import tqdm
tqdm.pandas(desc="dataset tokenization")

class VocabularyStoreHandler:

    @staticmethod
    def store_vocabularies(vocabulary_generator, voc_path):
        """Generates and stores a vocabulary to the specified path."""
        vocabulary = vocabulary_generator.generate()
        with open(voc_path, "w") as f:
            json.dump(vocabulary, f, indent=4)

        return vocabulary

    @staticmethod
    def load_vocabulary(voc_path):
        """Loads a vocabulary from the specified JSON file path."""
        if not os.path.exists(voc_path):
            raise FileNotFoundError(f"Can't find vocabulary file: {voc_path}")

        with open(voc_path, "r", encoding="utf-8") as f:
            return json.load(f)

class DatasetHandler:
    """Manages dataset loading, vocabulary generation, tokenization, and padding."""

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

        self.train = self.config_yaml['split_train']
        self.val = self.config_yaml['split_val']
        self.test = self.config_yaml['split_test']

        self.modified = False
        if self.config_yaml['first_time'] is True:
            self.modified = True

        self.codeTokenizer = None
        self.text_padding_handler = None
        self.englishTokenizer = None
        self.code_padding_handler = None

        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.keywords_path = os.path.join(self.root_dir, "data", "main_keywords.json")

        try:
            with open(self.keywords_path, "r", encoding="utf-8") as file:
                self.main_keywords = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            self.main_keywords = []

        self.yaml_path = None

    def load_dataset(self, split: Literal['train', 'val', 'test']):
        """Loads the tokenized dataset for the requested split, initializing it if necessary."""
        if self.modified:
            self.__save_config()
            print("Creation tokenized dataset...")
            self.create_dataset()
            self.config_yaml['first_time'] = False
        else:
            self.__load_vocabularies()

        if split == self.train:
            path = self.dataset_train_path
        elif split == self.val:
            path = self.dataset_val_path
        elif split == self.test:
            path = self.dataset_test_path
        else:
            raise ValueError("Invalid split")
        dataframe = pd.read_json(path, orient='records')
        return CodeDataset(torch.tensor(dataframe[self.config_yaml['dataset_x']], dtype=torch.long),
                           torch.tensor(dataframe[self.config_yaml['dataset_y']], dtype=torch.long))

    def set_yaml_path(self, yaml_path):
        """Sets the file path for the configuration YAML."""
        self.yaml_path = yaml_path

    def __save_config(self):
        """Saves the current configuration back to the YAML file."""
        if self.yaml_path is None:
            raise FileNotFoundError(f"Can't find yaml file: {self.yaml_path}")

        with open(self.yaml_path, 'w') as file:
            yaml.dump(self.config_yaml, file, default_flow_style=False, sort_keys=False)

    def __tokenize_pad(self, code: str, text: str) -> tuple[list[int], list[int]]:
        """Tokenizes and pads a code snippet and its corresponding text summary."""
        try:
            return (self.code_padding_handler.padding(self.codeTokenizer.tokenize(code)),
                    self.text_padding_handler.padding(
                        self.englishTokenizer.tokenize('[CLS]') + self.englishTokenizer.tokenize(text)
                        + self.englishTokenizer.tokenize('[SEP]')))
        except Exception:
            return (None, None)

    def create_dataset(self):
        """Generates vocabularies, tokenizes splits, and saves datasets to disk."""
        # create vocabulary
        dataset = load_dataset(self.dataset_link, self.dataset_config, split=self.train).to_pandas()

        snippets = dataset[self.config_yaml['dataset_x']]
        texts = dataset[self.config_yaml['dataset_y']]
        code_vocabulary_generator = PythonVocabularyGenerator(snippets)
        text_vocabulary_generator = EnglishVocabularyGenerator(texts)

        dict_code_vocabulary = VocabularyStoreHandler.store_vocabularies(code_vocabulary_generator, self.python_voc_path)
        dict_text_vocabulary = VocabularyStoreHandler.store_vocabularies(text_vocabulary_generator, self.english_voc_path)

        main_keywords = code_vocabulary_generator.get_main_keywords()
        self.main_keywords = main_keywords

        data_dir = os.path.join(self.root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        with open(self.keywords_path, "w", encoding="utf-8") as file:
            json.dump(self.main_keywords, file, indent=4, ensure_ascii=False)

        self.codeTokenizer = CodeTokenizer(dict_code_vocabulary, main_keywords)
        self.code_padding_handler = paddingHandler.PaddingHandler(self.max_code_len)

        self.englishTokenizer = EnglishTextTokenizer(dict_text_vocabulary)
        self.text_padding_handler = paddingHandler.PaddingHandler(1 + self.max_sum_len) #CLS + TEXT LEN

        # then create dataset
        splits_map = {
            self.dataset_train_path: self.train,
            self.dataset_val_path: self.val,
            self.dataset_test_path: self.test
        }

        for path, target in splits_map.items():
            dataset = load_dataset(self.dataset_link, self.dataset_config, split=target).to_pandas()

            tok_dataset = dataset.apply(
                lambda row: self.__tokenize_pad(row[self.config_yaml['dataset_x']], row[self.config_yaml['dataset_y']]),
                axis=1,
                result_type="expand"
            )
            tok_dataset.columns = [self.config_yaml['dataset_x'], self.config_yaml['dataset_y']]

            # delete none sample
            tok_dataset = tok_dataset.dropna()

            tok_dataset.to_json(path, orient="records", indent=2)

    def __load_vocabularies(self):
        """Loads pre-existing vocabularies and initializes tokenizers and padders."""
        dict_code_vocabulary = VocabularyStoreHandler.load_vocabulary(self.python_voc_path)
        dict_text_vocabulary = VocabularyStoreHandler.load_vocabulary(self.english_voc_path)

        self.codeTokenizer = CodeTokenizer(dict_code_vocabulary, self.main_keywords)
        self.code_padding_handler = paddingHandler.PaddingHandler(self.max_code_len)

        self.englishTokenizer = EnglishTextTokenizer(dict_text_vocabulary)
        self.text_padding_handler = paddingHandler.PaddingHandler(1 + self.max_sum_len)  # CLS + TEXT LEN