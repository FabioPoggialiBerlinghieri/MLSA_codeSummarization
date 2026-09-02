import itertools

import kagglehub
import pandas as pd
from pyarrow._dataset import Dataset

import SBTParse as a

class PythonVocabularyGenerator:
    def __init__(self, codes: list[str], len_max: int = 30000) -> None:
        self.codes = codes
        self.special_tokens = {
            "Name_UKN" : 0,              # Nomi di variabili o chiamate a funzione (se non l'hai rinominato in Variable)
            "Constant_UKN" : 1,          # Valori letterali (numeri come 2, 3.14, o stringhe come "Hello")
            "arg_UKN" : 2,               # Parametri passati a una funzione
            "FunctionDef_UKN" : 3,       # Nomi di funzioni definite
            "AsyncFunctionDef_UKN" : 4,  # Nomi di funzioni asincrone
            "ClassDef_UKN" : 5,          # Nomi di classi
            "Attribute_UKN" : 6,         # Attributi di oggetti (es. 'render' in self.render)
            "keyword_UKN" : 7,           # Argomenti passati per nome (kwargs)
            "alias_UKN" : 8,              # Nomi di moduli importati (es. import pandas as pd)
        }

        if len(self.special_tokens) > len_max:
            raise ValueError("Vocabulary max length must be at least ", len(self.special_tokens))

        self.LEN_MAX = len_max

    def generate(self) -> dict[str, int]:
        voc = []
        for code in self.codes:
            voc.extend(a.SBTParse().parse(code))
        occ_dict = dict.fromkeys(voc, 0)

        # Contiamo le occorrenze
        for v in voc:
            occ_dict[v] += 1

        # Ordinaimo e ci teniamo solo i più probabili
        occ_dict = dict(sorted(occ_dict.items(), key=lambda x: x[1], reverse=True))

        len_special_tokens = len(self.special_tokens)
        values = range(len_special_tokens + 1, len(occ_dict) + len_special_tokens + 1)

        occ_dict = dict(itertools.islice(occ_dict.items(), self.LEN_MAX - len_special_tokens - 1))

        voc = dict(zip(occ_dict.keys(), values))

        return self.special_tokens | {"UKN" : len_special_tokens} | voc

    def get_main_keywords(self) -> list[str]:
        main_keywords = []
        for word in self.special_tokens.keys():
            main_keywords.append(word.split("_")[0])
        return main_keywords

# Download latest version
#path = kagglehub.dataset_download("omduggineni/codesearchnet")#file = "~/.cache/kagglehub/datasets/omduggineni/codesearchnet/versions/500/python/python/final/jsonl/train/python_train_2.jsonl"
#snippets = pd.read_json(file, lines=True)
#snippets = snippets[:500]

