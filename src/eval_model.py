import argparse
import json
import evaluate as e
import paddingHandler
import tokenizer as tc
import torch
import yaml
from EDTransf import EDTransf
from paddingMask import PaddingMask
from vocabulary_generator import PythonVocabularyGenerator

bleu = e.load("bleu")
meteor = e.load("meteor")
rouge = e.load("rouge")

parser = argparse.ArgumentParser(description="TBD")
parser.add_argument('--checkpoint', type=str, required=True, help="Path best bleu weight file")
parser.add_argument('--split', type=str, default=None, help="Dataset split")

args = parser.parse_args()

checkpoint_path = args.checkpoint
split = args.split

saved_data = torch.load(checkpoint_path, map_location='cpu')
config = saved_data['config'] # dentro il check point ci deve essere il riferimento

model = EDTransf(embedding_dim=config['model']['embedding_dim'],
                 input_max_len=config['model']['input_max_len'],
                 input_vocabulary_size=config['model']['input_vocabulary_size'],
                 output_max_len=config['model']['output_max_len'],
                 output_vocabulary_size=config['model']['output_vocabulary_size'])

model.load_state_dict(saved_data['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()
dataset = CodeDatasetHandler(config).load_dataset(split) # mettere opzionali i campi max

with open(config['model']['python_voc_path'], "r") as f:
    code_vocabulary = json.load(f)

main_keywords = PythonVocabularyGenerator.get_main_keywords()
codeTokenizer = tc.CodeTokenizer(code_vocabulary, main_keywords)
code_padding_handler = paddingHandler.PaddingHandler(config['model']['max_code_len'])

with open(config['model']['english_voc_path'], "r") as f:
    english_vocabulary = json.load(f)

englishTokenizer = tc.EnglishTextTokenizer(english_vocabulary)
text_padding_handler = paddingHandler.PaddingHandler(config['model']['max_sum_len'])

# testing
generate_summs = []
target_sentences = []

for sample in dataset:
    code, target = sample
    with torch.no_grad():
        code = torch.tensor(code, device=device)
        mask = PaddingMask.generate_padding_mask(code).squeeze(1).to(device)
        cls = englishTokenizer.tokenize("[CLS]")[0]
        sep = englishTokenizer.tokenize("[SEP]")[0]
        summ_ids = model.predict(code, mask, cls, sep)

    summ_list = summ_ids[0].tolist()
    summ = englishTokenizer.detokenize(summ_list)
    summ_sentence = summ.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
    generate_summs.append(summ_sentence)

    target = englishTokenizer.detokenize(target)
    target_sentence = target.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
    target_sentences.append([target_sentence])

# compute metrix
blue_result = bleu.compute(predictions=generate_summs, references=target_sentences)
meteor_result = meteor.compute(predictions=generate_summs, references=target_sentences)
rouge_result = rouge.compute(predictions=generate_summs, references=target_sentences)

print(blue_result)
print(meteor_result)
print(rouge_result)