import ast
import codeTokenizer as tc
import vocabulary_generator as vg

# Dataset di prova: dimensioni piccole
from datasets import load_dataset
# Carica il dataset MBPP
dataset = load_dataset("google-research-datasets/mbpp", split="train")

snippets = dataset["code"]

v = vg.PythonVocabularyGenerator(snippets)
voc = v.generate()
main_keywords = v.get_main_keywords()

codeTokenizer = tc.CodeTokenizer(voc, main_keywords)

code1 = """
def saluto():
    return "ciao"
"""

result = codeTokenizer.tokenize(code1)
print(result)

chiavi = ""
for res in result:
    chiavi += " " + next((k for k, v in voc.items() if v == res), None)
print(chiavi)
