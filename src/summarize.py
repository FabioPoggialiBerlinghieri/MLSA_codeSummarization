import argparse
import torch
import os
from eval_model import ModelEvaluator
from datasetHandler import DatasetHandler
import set_seed as seed

if __name__ == "__main__":

    seed.set_deterministic_seed(42)

    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument('--input', type=str, required=True, help="Input code: a file .py or a directly a string code")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path best bleu weight file")
    parser.add_argument('--generate_mode', type=str, choices=['greedy', 'beam'], default='greedy',
                        help="Generate mode (default: greedy)")
    parser.add_argument('--beam_size', type=int, default=3, help="Beam size (default: 3)")

    args = parser.parse_args()

    code = ""
    if os.path.isfile(args.input):
        with open(args.input, "r") as f:
            code = f.read()
    else:
        code = args.input

    checkpoint_path = args.checkpoint
    generate_mode = args.generate_mode
    beam_size = args.beam_size

    saved_data = torch.load(checkpoint_path, map_location='cpu')

    dataset_handler = DatasetHandler(
        saved_data['config'],
        saved_data['config']['model']['max_code_len'],
        saved_data['config']['model']['max_text_len']
    )

    model_evaluator = ModelEvaluator(saved_data, dataset_handler, saved_data['config']['split_test'])
    summ_sentence = model_evaluator.summarize(code, generate_mode, beam_size)

    print("Input code:\n", code)
    print("Summarization sentence:\n", summ_sentence)
