import argparse
import evaluate as e
import rouge
import torch
from architecture.EDTransf import EDTransf
from data_processing.datasetHandler import DatasetHandler
from data_processing.paddingMask import PaddingMask
from utils import set_seed as seed

bleu = e.load("bleu")
rouge = e.load("rouge")

class ModelEvaluator:

    def __init__(self, saved_data, dataset_handler, split = None):
        self.saved_data = saved_data

        self.config = saved_data['config']

        self.model = EDTransf(embedding_dim=self.config['model']['embedding_dim'],
                         input_max_len=self.config['model']['max_code_len'],
                         input_vocabulary_size= self.config['model']['code_voc_len'],
                         output_max_len=self.config['model']['max_text_len'],
                         output_vocabulary_size=self.config['model']['text_voc_len'],
                         n_layers=self.config['model']['num_layers'],
                         n_heads=self.config['model']['num_heads'],
                         dropout=self.config['model']['dropout']
        )

        self.model.load_state_dict(saved_data['model_state_dict'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.dataset_handler = dataset_handler

        if split is not None:
            self.dataset = dataset_handler.load_dataset(split)

    def evaluate(self, batch_size, generate_mode, beam_size):
        """Evaluates the model on the loaded dataset split using BLEU and ROUGE metrics."""
        generate_summs = []
        target_sentences = []
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=False)

        for batch in test_loader:
            b_summ_sentences, b_target_sentences = SampleEvaluator.eval_sample(batch, self.model,
                                                                               self.dataset_handler.englishTokenizer,
                                                                               self.device, generate_mode, beam_size)
            generate_summs.extend(b_summ_sentences)
            target_sentences.extend([[t] for t in b_target_sentences])

        blue_result = bleu.compute(predictions=generate_summs, references=target_sentences)
        rouge_result = rouge.compute(predictions=generate_summs, references=target_sentences)

        return blue_result, rouge_result

    def summarize(self, code, generate_mode = "greedy", beam_size = 3):
        """Generates a summary for a single code snippet string."""
        self.model.eval()
        code = self.dataset_handler.codeTokenizer.tokenize(code)
        code = self.dataset_handler.code_padding_handler.padding(code)
        target = self.dataset_handler.englishTokenizer.tokenize("") # no
        target = self.dataset_handler.code_padding_handler.padding(target)

        code_tensor = torch.tensor([code])
        target_tensor = torch.tensor([target])
        sample_batch = (code_tensor, target_tensor)
        summ_sentence, _ = SampleEvaluator.eval_sample(sample_batch, self.model,
                                                                     self.dataset_handler.englishTokenizer, self.device, generate_mode, beam_size)
        return summ_sentence[0]

class SampleEvaluator:
    """Helper class to perform batch generation and text post-processing (detokenization)."""

    @staticmethod
    def eval_sample(sample_batch, model, tokenizer, device, generate_mode = "greedy", beam_size = 3) -> tuple[list[str], list[str]]:
        code, target = sample_batch
        with torch.no_grad():
            code = code.to(device)
            mask = PaddingMask.generate_padding_mask(code).to(device)
            cls = tokenizer.tokenize("[CLS]")[0]
            sep = tokenizer.tokenize("[SEP]")[0]
            pad = tokenizer.tokenize("[PAD]")[0]

            if generate_mode == "greedy":
                summ_ids = model.predict(code, mask, cls, sep, pad)
            if generate_mode == "beam":
                summ_ids = model.predict_bs(code, mask, cls, sep, pad, beam_size)

        summ_sentences = []
        target_sentences = []

        # for each batch size
        batch_size = code.size(0)
        for i in range(batch_size):
            # process predicted summary
            summ_list = summ_ids[i].tolist()
            summ = tokenizer.detokenize(summ_list)
            summ_sentence = summ.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
            summ_sentences.append(summ_sentence)

            # process ground truth target
            target_list = target[i].tolist()
            target_str = tokenizer.detokenize(target_list)
            target_sentence = target_str.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").strip()
            target_sentences.append(target_sentence)

        return summ_sentences, target_sentences


if __name__ == "__main__":

    seed.set_deterministic_seed(42)

    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path best bleu weight file")
    parser.add_argument('--split', type=str, default='test', help="Dataset split")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--generate_mode', type=str, choices=['greedy', 'beam'], default='greedy',
                        help="Generate mode (default: greedy)")
    parser.add_argument('--beam_size', type=int, default=3, help="Beam size (default: 3)")


    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    split = args.split
    batch_size = args.batch_size
    generate_mode = args.generate_mode
    beam_size = args.beam_size

    saved_data = torch.load(checkpoint_path, map_location='cpu')

    dataset_handler = DatasetHandler(
        saved_data['config'],
        saved_data['config']['model']['max_code_len'],
        saved_data['config']['model']['max_text_len']
    )

    if split == "val":
        split = saved_data['config']['split_val']
    if split == "test":
        split = saved_data['config']['split_test']

    model_evaluator = ModelEvaluator(saved_data, dataset_handler, split)
    bleu_result, rouge_result = model_evaluator.evaluate(batch_size, generate_mode, beam_size)

    print(bleu_result)
    print(rouge_result)