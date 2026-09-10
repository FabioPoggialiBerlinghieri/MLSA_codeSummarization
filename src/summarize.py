import json

import torch

import paddingHandler
from EDTransf import EDTransf
from paddingMask import PaddingMask
from vocabulary_generator import PythonVocabularyGenerator
import tokenizer as tc

device = "cpu"

code = """
x = x + 1
"""

with open("../data/code_vocab.json", "r") as f:
    code_vocabulary = json.load(f)

main_keywords = PythonVocabularyGenerator.get_main_keywords()
codeTokenizer = tc.CodeTokenizer(code_vocabulary, main_keywords)
code_padding_handler = paddingHandler.PaddingHandler(512)

with open("../data/english_vocab.json", "r") as f:
    english_vocabulary = json.load(f)

englishTokenizer = tc.EnglishTextTokenizer(english_vocabulary)
text_padding_handler = paddingHandler.PaddingHandler(128)

# tokenization and padding
code = code_padding_handler.padding(codeTokenizer.tokenize(code))

optimusPy = 3 # caricarlo con i pesi da capire ocme si fa

optimusPy.eval()
with torch.no_grad():
    code = torch.tensor(code, device=device)
    mask = PaddingMask.generate_padding_mask(code).squeeze(1).to(device)
    cls = englishTokenizer.tokenize("[CLS]")[0]
    sep = englishTokenizer.tokenize("[SEP]")[0]
    summ_ids = optimusPy.predict(code, mask, cls, sep)

summ_list = summ_ids[0].tolist()
summ = englishTokenizer.detokenize(summ_list)
print("E il nostro primo commento di x = x + 1 e':", summ)