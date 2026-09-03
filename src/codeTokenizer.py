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
        wrong_elements = []

        for main_keyword in self.main_keywords:
            if (main_keyword + "_" + self.uknown_identifier) not in self.vocabulary.keys():
                wrong_elements.append(main_keyword)

        if wrong_elements:
            raise InvalidMainKeywordsException(f"missmatch between main_keyword {wrong_elements} and vocabulary")

    def __check_unkownidentifier(self) -> None:
        if self.uknown_identifier not in self.vocabulary.keys():
            raise InvalidUknownIdentifierException("uknown identifier must be a key in the vocabulary")

    def tokenize(self, code: str) -> List[int]:
        """ Tokenize an indented Python source code string into a list of tokens """
        sbt = a.SBTParse().parse(code)
        return self.word2idx(sbt)

    def word2idx(self, words: List[str]) -> List[int]:
        """ Translate a list of words into a list of indices (tokens) """
        tokens = []
        for (word, i) in zip(words, range(len(words))):
            # if the word is not in the vocabulary, the word is replaced with a sepcial tokne
            # es: fun_uknown ---> FunctionDef_UNK
            if word not in self.vocabulary:
                if i > 1 and words[i-2] in self.main_keywords:
                    tokens.append(self.vocabulary[words[i-2] + "_" + self.uknown_identifier])
                elif i > 3 and words[i-4] in self.main_keywords and word == words[i - 2]:
                    tokens.append(self.vocabulary[words[i-4] + "_" + self.uknown_identifier])
                else:
                    tokens.append(self.vocabulary[self.uknown_identifier])
            else:
                tokens.append(self.vocabulary[word])
        return tokens
