import itertools
import SBTParse as a
import nltk
from abc import ABC, abstractmethod

class VocabularyGenerator(ABC):

    def __init__(self, codes: list[str], len_max: int, special_tokens: dict[str, int] = {}) -> None:
        self.codes = codes

        self.special_tokens = special_tokens

        if len(self.special_tokens) > len_max:
            raise ValueError(f"Vocabulary max length must be at least {len(self.special_tokens) + 1}")

        self.LEN_MAX = len_max

    def generate_with_strategy(self, tokenizer_fn) -> dict[str, int]:
        voc = []
        for code in self.codes:
            voc.extend(tokenizer_fn(code))
        occ_dict = dict.fromkeys(voc, 0)

        # Contiamo le occorrenze
        for v in voc:
            occ_dict[v] += 1

        # Ordinaimo e ci teniamo solo i più probabili
        occ_dict = dict(sorted(occ_dict.items(), key=lambda x: x[1], reverse=True))

        len_special_tokens = len(self.special_tokens)
        values = range(len_special_tokens + 2, len(occ_dict) + len_special_tokens + 2)

        occ_dict = dict(itertools.islice(occ_dict.items(), self.LEN_MAX - len_special_tokens - 2))

        voc = dict(zip(occ_dict.keys(), values))

        return {"[PAD]": 0, "[UNK]": 1} | self.special_tokens | voc

    @abstractmethod
    def generate(self) -> dict[str, int]:
        pass

class PythonVocabularyGenerator(VocabularyGenerator):

    def __init__(self, codes: list[str], len_max: int = 30000) -> None:
        super().__init__(codes, len_max, {
            "Name_UNK" : 2,              # Nomi di variabili o chiamate a funzione (se non l'hai rinominato in Variable)
            "Constant_UNK" : 3,          # Valori letterali (numeri come 2, 3.14, o stringhe come "Hello")
            "arg_UNK" : 4,               # Parametri passati a una funzione
            "FunctionDef_UNK" : 5,       # Nomi di funzioni definite
            "AsyncFunctionDef_UNK" : 6,  # Nomi di funzioni asincrone
            "ClassDef_UNK" : 7,          # Nomi di classi
            "Attribute_UNK" : 8,         # Attributi di oggetti (es. 'render' in self.render)
            "keyword_UNK" : 9,           # Argomenti passati per nome (kwargs)
            "alias_UNK" : 10,              # Nomi di moduli importati (es. import pandas as pd)
        })

    def generate(self) -> dict[str, int]:
        return super().generate_with_strategy(lambda x: a.SBTParse().parse(x))

    def get_main_keywords(self) -> list[str]:
        main_keywords = []
        for word in self.special_tokens.keys():
            main_keywords.append(word.split("_")[0])
        return main_keywords

class EnglishVocabularyGenerator(VocabularyGenerator):

    def __init__(self, codes: list[str], len_max: int = 30000) -> None:
        super().__init__(codes, len_max, {
            "[CLS]" : 2,
            "[SEP]" : 3,
            "[MASK]": 4
        })

    def generate(self) -> dict[str, int]:
        nltk.download('punkt_tab')
        return super().generate_with_strategy(lambda x: nltk.word_tokenize(x))
