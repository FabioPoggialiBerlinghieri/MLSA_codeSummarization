import ast
from typing import Dict, List
import SBTParse as a

class InvalidUknownIdentifierException(Exception):
    pass

class InvalidMainKeywordsException(Exception):
    pass

class CodeTokenizer:

    def __init__(self, vocabulary: Dict[str, int] = None, main_keywords: List[str] = None, uknown_identifier: str = "UKN") -> None:
        self.vocabulary = vocabulary
        self.main_keywords = main_keywords
        self.uknown_identifier = uknown_identifier

        self.__check_unkownidentifier()
        self.__check_missmatch_vocabulary_main_keywords()

    def __check_missmatch_vocabulary_main_keywords(self) -> None:
        for main_keyword in self.main_keywords:
            if (main_keyword + "_" + self.uknown_identifier) not in self.vocabulary.keys():
                raise InvalidMainKeywordsException("missmatch between main_keyword ", main_keyword, " and vocabulary")

    def __check_unkownidentifier(self):
        if self.uknown_identifier not in self.vocabulary.keys():
            raise InvalidUknownIdentifierException("uknown identifier must be a key in the vocabulary")

    def set_vocabulary(self, vocabulary: Dict[str, int]) -> None:
        self.vocabulary = vocabulary
        self.__check_missmatch_vocabulary_main_keywords()
        self.__check_unkownidentifier()

    def set_main_keywords(self, main_keywords: List[str]) -> None:
        self.main_keywords = main_keywords
        self.__check_missmatch_vocabulary_main_keywords()

    def set_uknown_identifier(self, uknown_identifier: str) -> None:
        self.uknown_identifier = uknown_identifier
        self.__check_unkownidentifier()

    def tokenize(self, text: str) -> List[int]:
        sbt = a.SBTParse().parse(text)
        return self.__word2idx(sbt)

    def __word2idx(self, words: List[str]) -> List[int]:
        tokens = []
        for (word, i) in zip(words, range(len(words))):
            # se la parola non è nel vocabolario, si sostituisce con un token speciale
            # es: nome_fun_sconosciuto ---> FunctionDef_UNK
            if word not in self.vocabulary:
                if words[i-2] in self.main_keywords:
                    tokens.append(self.vocabulary[words[i-2] + "_" + self.uknown_identifier])
                elif words[i-4] in self.main_keywords and word == words[i - 2]:
                    tokens.append(self.vocabulary[words[i-4] + "_" + self.uknown_identifier])
                else:
                    tokens.append(self.vocabulary[self.uknown_identifier])
            else:
                tokens.append(self.vocabulary[word])
        return tokens

