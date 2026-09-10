import json
import paddingHandler
import tokenizer as tc
import vocabulary_generator as vg
from datasets import load_dataset

def tokenize_pad(code: str, text: str) -> tuple[list[int], list[int]]:

    return (code_padding_handler.padding(codeTokenizer.tokenize(code)),
            text_padding_handler.padding(englishTokenizer.tokenize('[CLS]') + englishTokenizer.tokenize(text)
                                        + englishTokenizer.tokenize('[SEP]')))

# Dataset di prova: dimensioni piccole
# Carica il dataset MBPP
dataset = load_dataset("google-research-datasets/mbpp", split="train").to_pandas()

snippets = dataset["code"]
texts = dataset["text"]

python_voc_generator = vg.PythonVocabularyGenerator(snippets)
code_vocabulary = python_voc_generator.generate()

with open("../data/code_vocab.json", "w") as f:
    json.dump(code_vocabulary, f, indent=4)

main_keywords = python_voc_generator.get_main_keywords()
codeTokenizer = tc.CodeTokenizer(code_vocabulary, main_keywords)
code_padding_handler = paddingHandler.PaddingHandler(512)

english_voc_generator = vg.EnglishVocabularyGenerator(texts)
english_vocabulary = english_voc_generator.generate()

with open("../data/english_vocab.json", "w") as f:
    json.dump(english_vocabulary, f, indent=4)

englishTokenizer = tc.EnglishTextTokenizer(english_vocabulary)
text_padding_handler = paddingHandler.PaddingHandler(1 + 128) # CLS + label

tok_dataset = dataset.apply(
    lambda row: tokenize_pad(row["code"], row["text"]),
    axis=1,
    result_type="expand"
)
tok_dataset.columns = ["code", "text"]

tok_dataset.to_json("../data/dataset_train.json", orient="records", indent=2)


dataset = load_dataset("google-research-datasets/mbpp", split="validation").to_pandas()

tok_dataset = dataset.apply(
    lambda row: tokenize_pad(row["code"], row["text"]),
    axis=1,
    result_type="expand"
)
tok_dataset.columns = ["code", "text"]

tok_dataset.to_json("../data/dataset_validation.json", orient="records", indent=2)
