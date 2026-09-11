import argparse
import json
import torch
import os
import paddingHandler
from EDTransf import EDTransf
from eval_model import ModelEvaluator
from main import DatasetHandler
from paddingMask import PaddingMask
from vocabulary_generator import PythonVocabularyGenerator
import tokenizer as tc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument('--input', type=str, required=True, help="Input code: a file .py or a directly a string code")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path best bleu weight file")

    args = parser.parse_args()

    code = ""
    if os.path.isfile(args.input):
        with open(args.input, "r") as f:
            code = f.read()
    else:
        code = args.input

    checkpoint_path = args.checkpoint

    saved_data = torch.load(checkpoint_path, map_location='cpu')

    dataset_handler = DatasetHandler(
        saved_data['config'],
        saved_data['config']['model']['max_code_len'],
        saved_data['config']['model']['max_text_len']
    )

    model_evaluator = ModelEvaluator(saved_data, dataset_handler, "test")
    summ_sentence = model_evaluator.summarize(code)

    print("Input code:\n", code)
    print("Summarization sentence:\n", summ_sentence)
